#run_deepshell.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_samples,
)
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Deep GMM with TURTLE's Method")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to data directory"
    )
    parser.add_argument(
        "--representation_types",
        nargs="+",
        default=["clipvitL14", "dinov2"],
        help="Types of representations to use",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=150, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--min_clusters",
        type=int,
        default=2,
        help="Minimum number of clusters to consider",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=12,
        help="Maximum number of clusters to consider",
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU if available"
    )
    parser.add_argument(
        "--pca_components",
        type=float,
        default=None,
        help="Number of PCA components to retain (float for variance ratio, int for components)",
    )
    args = parser.parse_args()
    return args


def run_deep_gmm_turtle(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load and preprocess training data
    X_trains = []
    scalers = []  # List to store scalers for each representation type
    for rep_type in args.representation_types:
        X_train = np.load(
            os.path.join(
                args.data_path, f"representations/{rep_type}/cifar10_train.npy"
            )
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_trains.append(X_train)
        scalers.append(scaler)  # Store the fitted scaler

    X_train_all = np.concatenate(X_trains, axis=1)
    print(f"Combined training data shape: {X_train_all.shape}")

    # Load and preprocess validation data
    X_vals = []
    for idx, rep_type in enumerate(args.representation_types):
        X_val = np.load(
            os.path.join(
                args.data_path, f"representations/{rep_type}/cifar10_val.npy"
            )
        )
        scaler = scalers[idx]  # Retrieve the corresponding scaler
        X_val = scaler.transform(X_val)
        X_vals.append(X_val)

    X_val_all = np.concatenate(X_vals, axis=1)
    print(f"Combined validation data shape: {X_val_all.shape}")

    # Load validation labels (for evaluation only)
    y_true_val = np.load(
        os.path.join(args.data_path, "labels/cifar10_val_labels.npy")
    )

    # PCA for dimensionality reduction
    print("Applying PCA...")
    if args.pca_components is not None:
        if isinstance(args.pca_components, float):
            pca = PCA(n_components=args.pca_components, svd_solver="full")
        elif isinstance(args.pca_components, int):
            pca = PCA(n_components=args.pca_components)
        else:
            raise ValueError(
                "Invalid pca_components type. Must be float or int."
            )
        X_train_reduced = pca.fit_transform(X_train_all)
        X_val_reduced = pca.transform(X_val_all)
        print(f"Reduced training data shape: {X_train_reduced.shape}")
        print(f"Reduced validation data shape: {X_val_reduced.shape}")
    else:
        X_train_reduced = X_train_all  # No PCA
        X_val_reduced = X_val_all
        print("No PCA applied. Using original data.")

    # Prepare training data
    X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Define neural network (Autoencoder)
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1000),
                nn.ReLU(),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 200),
                nn.ReLU(),
                nn.Linear(200, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 200),
                nn.ReLU(),
                nn.Linear(200, 500),
                nn.ReLU(),
                nn.Linear(500, 1000),
                nn.ReLU(),
                nn.Linear(1000, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z

    # Initialize model, optimizer, and loss functions
    input_dim = X_train_reduced.shape[1]
    latent_dim = 50  # Adjust as needed
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    reconstruction_loss_fn = nn.MSELoss()

    # Training loop
    print("Starting training...")
    training_losses = []
    for epoch in range(args.max_epochs):
        model.train()
        epoch_recon_loss = 0.0
        for batch in train_dataloader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            x_recon, z = model(x_batch)
            recon_loss = reconstruction_loss_fn(x_recon, x_batch)
            recon_loss.backward()
            optimizer.step()
            epoch_recon_loss += recon_loss.item() * x_batch.size(0)

        epoch_recon_loss /= len(train_dataloader.dataset)
        training_losses.append(epoch_recon_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{args.max_epochs}, Reconstruction Loss: {epoch_recon_loss:.4f}"
            )

    # After training, plot and save training loss
    plt.figure()
    plt.plot(range(1, args.max_epochs + 1), training_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Training Loss")
    plt.savefig(
        os.path.join(args.data_path, "results", "training_loss.png")
    )
    plt.close()

    # After training, get latent representations for training data
    model.eval()
    with torch.no_grad():
        X_train_latent = []
        for batch in DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        ):
            x_batch = batch[0].to(device)
            _, z = model(x_batch)
            X_train_latent.append(z.cpu().numpy())
        X_train_latent = np.concatenate(X_train_latent, axis=0)

    # Get latent representations for validation data
    X_val_tensor = torch.tensor(X_val_reduced, dtype=torch.float32)
    val_dataset = TensorDataset(X_val_tensor)
    with torch.no_grad():
        X_val_latent = []
        for batch in DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        ):
            x_batch = batch[0].to(device)
            _, z = model(x_batch)
            X_val_latent.append(z.cpu().numpy())
        X_val_latent = np.concatenate(X_val_latent, axis=0)

    # Determine optimal number of clusters using training data
    n_clusters_list = list(range(args.min_clusters, args.max_clusters + 1))
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    gmm_models = []

    print("Evaluating clustering performance on training data for different number of clusters...")
    for n_clusters in n_clusters_list:
        print(f"Evaluating GMM with {n_clusters} clusters...")
        gmm = GaussianMixture(
            n_components=n_clusters, covariance_type="full", random_state=42
        )
        gmm.fit(X_train_latent)
        labels_train = gmm.predict(X_train_latent)
        silhouette_avg = silhouette_score(X_train_latent, labels_train)
        ch_score = calinski_harabasz_score(X_train_latent, labels_train)
        db_score = davies_bouldin_score(X_train_latent, labels_train)
        silhouette_scores.append(silhouette_avg)
        ch_scores.append(ch_score)
        db_scores.append(db_score)
        gmm_models.append(gmm)
        print(
            f"Silhouette Score: {silhouette_avg:.4f}, Calinski-Harabasz Index: {ch_score:.2f}, Davies-Bouldin Index: {db_score:.4f}"
        )

    # Plot evaluation metrics
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(n_clusters_list, silhouette_scores, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score (Training Data)")

    plt.subplot(1, 3, 2)
    plt.plot(n_clusters_list, ch_scores, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Calinski-Harabasz Index")
    plt.title("Calinski-Harabasz Index (Training Data)")

    plt.subplot(1, 3, 3)
    plt.plot(n_clusters_list, db_scores, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("Davies-Bouldin Index (Training Data)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            args.data_path, "results", "clustering_evaluation_metrics_training.png"
        )
    )
    plt.close()

    # Combine metrics to select top 5 cluster numbers
    # For Silhouette Score and Calinski-Harabasz Index, higher is better
    # For Davies-Bouldin Index, lower is better

    # Normalize the metrics
    scaler_metrics = MinMaxScaler()
    sil_scores_scaled = scaler_metrics.fit_transform(
        np.array(silhouette_scores).reshape(-1, 1)
    ).flatten()
    ch_scores_scaled = scaler_metrics.fit_transform(
        np.array(ch_scores).reshape(-1, 1)
    ).flatten()
    db_scores_scaled = scaler_metrics.fit_transform(
        -np.array(db_scores).reshape(-1, 1)
    ).flatten()  # Negative because lower is better

    # Compute combined score
    combined_scores = sil_scores_scaled + ch_scores_scaled + db_scores_scaled

    # Get top 5 cluster numbers
    top_indices = np.argsort(combined_scores)[-5:]  # Get indices of top 5 scores
    top_cluster_numbers = [n_clusters_list[i] for i in top_indices]

    print(
        f"Top 5 cluster numbers based on combined metrics: {top_cluster_numbers}"
    )

    # Evaluate models for top cluster numbers on validation data
    metrics_list = []  # To store metrics for saving later

    for n_clusters in top_cluster_numbers:
        index = n_clusters_list.index(n_clusters)
        gmm = gmm_models[index]
        labels_val = gmm.predict(X_val_latent)

        # Evaluate clustering performance on validation data
        nmi_score = normalized_mutual_info_score(y_true_val, labels_val)
        ari_score = adjusted_rand_score(y_true_val, labels_val)
        silhouette_avg = silhouette_score(X_val_latent, labels_val)
        ch_score = calinski_harabasz_score(X_val_latent, labels_val)
        db_score = davies_bouldin_score(X_val_latent, labels_val)

        # Calculate clustering accuracy
        cluster_acc = clustering_accuracy(y_true_val, labels_val)

        print(f"\nResults for {n_clusters} clusters on validation data:")
        print(f"Clustering Accuracy: {cluster_acc:.4f}")
        print(f"NMI Score: {nmi_score:.4f}")
        print(f"ARI Score: {ari_score:.4f}")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Calinski-Harabasz Index: {ch_score:.2f}")
        print(f"Davies-Bouldin Index: {db_score:.4f}")

        # Save metrics to list
        metrics_list.append(
            {
                "n_clusters": n_clusters,
                "cluster_acc": cluster_acc,
                "nmi_score": nmi_score,
                "ari_score": ari_score,
                "silhouette_score": silhouette_avg,
                "calinski_harabasz_index": ch_score,
                "davies_bouldin_index": db_score,
            }
        )

        # Visualization using validation data
        # t-SNE Visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_embedded = tsne.fit_transform(X_val_latent)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_embedded[:, 0], X_embedded[:, 1], c=labels_val, cmap="tab10", s=5
        )
        plt.title(f"t-SNE Visualization with {n_clusters} Clusters (Validation Data)")
        plt.savefig(
            os.path.join(
                args.data_path,
                "results",
                f"tsne_{n_clusters}_clusters_validation.png",
            )
        )
        plt.close()

        # PCA 2D Visualization
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_val_latent)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_pca_2d[:, 0],
            X_pca_2d[:, 1],
            c=labels_val,
            cmap="tab10",
            s=5,
        )
        plt.title(f"PCA 2D Visualization with {n_clusters} Clusters (Validation Data)")
        plt.savefig(
            os.path.join(
                args.data_path,
                "results",
                f"pca_2d_{n_clusters}_clusters_validation.png",
            )
        )
        plt.close()

        # Silhouette Plot
        # After calculating silhouette scores over a range of clusters
        plt.figure()
        plt.plot(n_clusters_list, silhouette_scores, label="Silhouette Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score across Clusters")
        plt.legend()
        plt.savefig(os.path.join(args.data_path, "results", "silhouette_score_line_plot.png"))
        plt.close()

        # Save results
        results_dir = os.path.join(
            args.data_path, "results", f"deepgmm_turtle_{n_clusters}_clusters"
        )
        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, "latent_space_validation.npy"), X_val_latent)
        np.save(
            os.path.join(results_dir, "cluster_assignments_validation.npy"), labels_val
        )
        print(f"Results saved to {results_dir}")

    # Save all metrics to a CSV file
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(
        os.path.join(args.data_path, "results", "clustering_metrics.csv"),
        index=False,
    )
    print(
        f"All metrics saved to {os.path.join(args.data_path, 'results', 'clustering_metrics.csv')}"
    )

    # Plot Clustering Accuracy vs. Number of Clusters
    plt.figure()
    plt.plot(
        [metrics["n_clusters"] for metrics in metrics_list],
        [metrics["cluster_acc"] for metrics in metrics_list],
        marker="o",
    )
    plt.xlabel("Number of clusters")
    plt.ylabel("Clustering Accuracy")
    plt.title("Clustering Accuracy (Validation Data)")
    plt.savefig(
        os.path.join(args.data_path, "results", "clustering_accuracy.png")
    )
    plt.close()

    # Correlation Matrix of Latent Features (Validation Data)
    corr_matrix = np.corrcoef(X_val_latent.T)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", square=True)
    plt.title("Correlation Matrix of Latent Features (Validation Data)")
    plt.savefig(
        os.path.join(args.data_path, "results", "correlation_matrix.png")
    )
    plt.close()

    # Dendrogram (Validation Data)
    # Use a subset of data for dendrogram if data is large
    sample_indices = np.random.choice(
        X_val_latent.shape[0], size=500, replace=False
    )
    X_sample = X_val_latent[sample_indices]

    linked = linkage(X_sample, "single")

    plt.figure(figsize=(10, 7))
    dendrogram(
        linked,
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=False,
    )
    plt.title("Hierarchical Clustering Dendrogram (Validation Data)")
    plt.savefig(
        os.path.join(args.data_path, "results", "dendrogram.png")
    )
    plt.close()


def clustering_accuracy(y_true, y_pred):
    # y_true and y_pred are numpy arrays
    from scipy.optimize import linear_sum_assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_row, ind_col = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) / y_pred.size


if __name__ == "__main__":
    args = parse_args()
    # Ensure the results directory exists
    os.makedirs(os.path.join(args.data_path, "results"), exist_ok=True)
    run_deep_gmm_turtle(args)
