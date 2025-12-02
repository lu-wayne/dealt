import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from utils import get_class_counts, compute_embeddings
from config import DEALTConfig
from sklearn.metrics.pairwise import euclidean_distances # DBSCAN uses distance metric

class LongTailDistributionDetector:
    """Detects explicit and implicit long-tail patterns."""
    def __init__(self, config: DEALTConfig, train_df: pd.DataFrame):
        self.config = config
        self.train_df = train_df
        self.class_counts = get_class_counts(train_df)
        self.classes = sorted(self.class_counts.keys())
        self.total_samples = len(train_df)
        self.avg_class_size = self.total_samples / len(self.classes)
        self.dataset_name = 'default_dataset' # Needs to be set based on input

    def set_dataset_name(self, name):
        """Set dataset name to load specific configs."""
        self.dataset_name = name

    def identify_explicit_tails(self):
        """Identifies explicit long-tail classes based on counts."""
        explicit_tail_classes = []
        threshold_count = max(
            self.config.K_EXPLICIT_THRESHOLD_MIN_COUNT,
            np.percentile(list(self.class_counts.values()), self.config.K_EXPLICIT_THRESHOLD_PERCENTILE)
        )

        for cls, count in self.class_counts.items():
            if count < threshold_count:
                explicit_tail_classes.append(cls)

        return explicit_tail_classes

    def identify_implicit_tails(self):
        """Identifies implicit long-tail clusters within each class."""
        implicit_tail_clusters = {} # {class: [list of sample indices in original df]}

        # Calculate implicit threshold based on average class size if needed
        implicit_threshold_count_dynamic = max(
             self.config.K_IMPLICIT_THRESHOLD_COUNT,
             int(self.avg_class_size * (self.config.K_IMPLICIT_THRESHOLD_PERCENTAGE_AVG / 100))
        )

        for cls in tqdm(self.classes, desc="Identifying implicit tails"):
            class_df = self.train_df[self.train_df[self.config.LABEL_COLUMN] == cls].reset_index(drop=True)
            if len(class_df) < self.config.DBSCAN_MIN_SAMPLES:
                continue # Cannot cluster with too few samples

            texts = class_df[self.config.TEXT_COLUMN].tolist()
            embeddings = compute_embeddings(texts)

            if embeddings.shape[0] == 0:
                 continue

            # Get dataset-specific eps or use default
            dataset_config = self.config.get_dataset_config(self.dataset_name)
            dbscan_eps = dataset_config['dbscan_eps']

            # Use cosine distance if normalized embeddings are used (1 - cosine_similarity)
            # DBSCAN can take a precomputed distance matrix
            # dist_matrix = 1 - cosine_similarity(embeddings)
            # db = DBSCAN(eps=dbscan_eps, min_samples=self.config.DBSCAN_MIN_SAMPLES, metric='precomputed').fit(dist_matrix)

            # Or use Euclidean distance on normalized embeddings (common for Sentence-BERT)
            db = DBSCAN(eps=dbscan_eps, min_samples=self.config.DBSCAN_MIN_SAMPLES).fit(embeddings)


            labels = db.labels_
            unique_labels = set(labels)

            for k in unique_labels:
                if k == -1: # Noise points
                    cluster_indices = class_df.index[labels == k].tolist()
                    continue

                cluster_indices_in_class_df = class_df.index[labels == k].tolist()

                # Identify implicit tail if cluster size is below threshold
                if len(cluster_indices_in_class_df) < implicit_threshold_count_dynamic:
                    # Map indices back to original train_df indices
                    original_indices = self.train_df[
                        (self.train_df[self.config.LABEL_COLUMN] == cls)
                    ].iloc[cluster_indices_in_class_df].index.tolist()

                    if cls not in implicit_tail_clusters:
                        implicit_tail_clusters[cls] = []
                    implicit_tail_clusters[cls].extend(original_indices)

        # Remove duplicates if any (shouldn't be if adding original_indices list)
        for cls in implicit_tail_clusters:
             implicit_tail_clusters[cls] = list(set(implicit_tail_clusters[cls]))

        return implicit_tail_clusters # Format: {class: [list of original df indices]}

    def select_representative_samples(self, implicit_tail_clusters):
        """Selects representative samples from implicit tail clusters."""
        representative_samples = pd.DataFrame() # Store sample dataframes

        for cls, indices in implicit_tail_clusters.items():
            cluster_df = self.train_df.loc[indices].reset_index(drop=True)
            if cluster_df.empty:
                continue

            # Select k_rep representative samples
            if len(cluster_df) <= self.config.K_REP:
                 representative_samples = pd.concat([representative_samples, cluster_df])
            else:
                # Calculate centroid and find sample closest to it
                texts = cluster_df[self.config.TEXT_COLUMN].tolist()
                embeddings = compute_embeddings(texts)
                if embeddings.shape[0] == 0:
                    continue

                centroid = np.mean(embeddings, axis=0)
                # Compute Euclidean distance (or cosine distance if preferred)
                distances = euclidean_distances(embeddings, centroid.reshape(1, -1)).flatten()
                closest_indices_in_cluster_df = np.argsort(distances)[:self.config.K_REP]
                representative_samples = pd.concat([representative_samples, cluster_df.iloc[closest_indices_in_cluster_df]])

        return representative_samples.reset_index(drop=True) # DataFrame of representative samples

    def identify_target_samples(self):
        """Identifies all samples from explicit and implicit tails for potential augmentation."""
        explicit_tail_classes = self.identify_explicit_tails()
        implicit_tail_clusters = self.identify_implicit_tails()

        # Get samples from explicit tail classes
        explicit_tail_samples = self.train_df[self.train_df[self.config.LABEL_COLUMN].isin(explicit_tail_classes)]

        # Get representative samples from implicit tail clusters
        implicit_representative_samples = self.select_representative_samples(implicit_tail_clusters)

        # Combine and get unique samples (sample might be in both explicit and implicit)
        starget_df = pd.concat([explicit_tail_samples, implicit_representative_samples]).drop_duplicates(subset=[self.config.TEXT_COLUMN, self.config.LABEL_COLUMN]).reset_index(drop=True)

        # Store info about which samples belong to implicit clusters
        implicit_cluster_info = {}
        for cls, indices in implicit_tail_clusters.items():
            implicit_cluster_info[cls] = self.train_df.loc[indices].to_dict('records') # Store list of sample dicts

        return starget_df, explicit_tail_classes, implicit_cluster_info

