# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import os
import pickle
import argparse
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE

def perform_kmeans_clustering(num_clusters, use_resnet=False, use_tsne=False):

    # check only one of use_resnet and use_tsne is True
    assert not (use_resnet and use_tsne), "Only one of use_resnet and use_tsne"

    # global variables
    if use_resnet:
        save_folder = f"results_{num_clusters}_resnet"
        model_path = f"resnet_model_{num_clusters}.pkl"
    elif use_tsne:
        save_folder = f"results_{num_clusters}_tsne"
        model_path = f"tsne_model_{num_clusters}.pkl"
    else:
        save_folder = f"results_{num_clusters}"
        model_path = f"kmeans_model_{num_clusters}.pkl"
    os.makedirs(save_folder, exist_ok=True)

    # Load the Wholesale Customer Dataset from UCI repository
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # load data
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # class index to name mapping
    class_idx_to_name = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    # get train and test data
    c, h, w = trainset[0][0].shape
    train_data = np.zeros((len(trainset), c * h * w))
    train_category = np.zeros((len(trainset)))
    for i in range(len(trainset)):
        train_data[i] = trainset[i][0].reshape(c * h * w).numpy()
        train_category[i] = trainset[i][1]
    print("train_data shape:", train_data.shape, "train_category shape:", train_category.shape)

    test_data = np.zeros((len(testset), c * h * w))
    test_category = np.zeros((len(testset)))
    for i in range(len(testset)):
        test_data[i] = testset[i][0].reshape(c * h * w).numpy()
        test_category[i] = testset[i][1]
    print("test_data shape:", test_data.shape, "test_category shape:", test_category.shape)

    if use_resnet:
        print("Using ResNet-50 features for clustering...")
        # use resnet features
        resnet_50 = torchvision.models.resnet50(pretrained=True)
        resnet_50.fc = torch.nn.Identity()
        resnet_50 = resnet_50.cuda()
        resnet_50.eval()

        # get train and test data
        train_data = torch.tensor(train_data).cuda().float().view(-1, c, h, w)
        results = []
        for i in range(5):
            results.append(resnet_50(train_data[i*10000:(i+1)*10000]).detach().cpu().numpy())
        train_data = np.concatenate(results, axis=0)

        test_data = torch.tensor(test_data).cuda().float().view(-1, c, h, w)
        test_data = resnet_50(test_data).detach().cpu().numpy()

    if use_tsne:
        # Reduce the dimensionality of the data using t-SNE
        tsne = TSNE(n_components=2, random_state=42, verbose=2)
        train_data = tsne.fit_transform(train_data)

    # Check if the model already exists
    if os.path.exists(model_path):
        # Load the model
        with open(model_path, "rb") as file:
            kmeans = pickle.load(file)
        print(f"Model loaded from {model_path}")
    else:
        # Perform K-Means clustering with n_init=10, and max_iter=300
        kmeans = KMeans(
            n_clusters=num_clusters, random_state=42, n_init=5, max_iter=300, verbose=2
        ).fit(train_data)

        # Save the model to disk
        with open(model_path, "wb") as file:
            pickle.dump(kmeans, file)
        print(f"Model saved to {model_path}")


    # visualize the clusters if TSNE is used
    if use_tsne:
        train_pred = kmeans.predict(train_data)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=train_data[:, 0], y=train_data[:, 1], hue=train_pred, palette='deep', s=100)
        plt.title(f"K-Means Clustering (K={num_clusters}) - t-SNE")
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.grid(True)
        plt.savefig(f"cifar10_tsne_{num_clusters}.png")

    # print silhouette score
    silhouette_avg = silhouette_score(train_data, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    # calculate NMIs on train data
    train_labels = kmeans.predict(train_data)
    nmi = normalized_mutual_info_score(train_category, train_labels)
    print(f"Normalized Mutual Information Score: {nmi:.4f}")

    # calculate ARIs on train data
    train_labels = kmeans.predict(train_data)
    ari = adjusted_rand_score(train_category, train_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    if not use_resnet and not use_tsne:
        # visualize the clusters as images
        centers = kmeans.cluster_centers_
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        for i, ax in enumerate(axes.flat):
            # unnormalize the image
            image = centers[i].reshape(c, h, w).transpose(1, 2, 0)
            image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"Cluster {i}")
        fig.tight_layout()
        fig.savefig(f"cifar10_centers_{num_clusters}.png")

        # top 10 closest images to each center in test_data
        # save each cluster images to a folder individually
        closest_images = []
        for i in range(num_clusters):
            distances = np.linalg.norm(train_data - centers[i], axis=1)
            closest_indices = np.argsort(distances)[:10]
            closest_images.append(closest_indices)

            # combine the images horizontally
            images = []
            for j in range(10):
                image = train_data[closest_indices[j]].reshape(c, h, w).transpose(1, 2, 0)
                image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
                image = np.clip(image, 0, 1)
                images.append(image)
            images = np.concatenate(images, axis=1)
            
            # add the cluster name
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.imshow(images)
            ax.axis("off")
            ax.set_title(f"Cluster {i}")
            plt.imsave(f"{save_folder}/cluster_{i}.png", images)
            
        
    return kmeans, train_data, train_category, test_data, test_category, class_idx_to_name, c, h, w


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="K-Means Clustering on CIFAR-10 dataset")
    parser.add_argument(
        "--use_resnet", action="store_true", help="Use ResNet-50 features for clustering"
    )
    parser.add_argument(
        "--use_tsne", action="store_true", help="Use t-SNE for visualization"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=10, help="Number of clusters for K-Means"
    )
    args = parser.parse_args()

    kmeans, train_data, train_category, test_data, test_category, class_idx_to_name, c, h, w = perform_kmeans_clustering(args.num_clusters, args.use_resnet, args.use_tsne)

# # Predict the clusters
# centers = kmeans.cluster_centers_

# # visualize centers
# fig, axes = plt.subplots(2, 5, figsize=(20, 10))
# for i, ax in enumerate(axes.flat):
#     # unnormalize the image
#     image = centers[i].reshape(c, h, w).transpose(1, 2, 0)
#     image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
#     image = np.clip(image, 0, 1)
#     ax.imshow(image)
#     ax.axis("off")
#     ax.set_title(f"Cluster {i}")
# fig.tight_layout()
# fig.savefig("cifar10_centers.png")

# # top 10 closest images to each center in test_data
# closest_images = []
# for i in range(10):
#     distances = np.linalg.norm(train_data - centers[i], axis=1)
#     closest_indices = np.argsort(distances)[:10]
#     closest_images.append(closest_indices)

# # visualize closest images and their GT labels
# fig, axes = plt.subplots(
#     11, 10, figsize=(20, 44)
# )  # Adjusted figsize to accommodate cluster names
# for i in range(num_clusters):
#     for j in range(10):
#         ax = axes[i, j]
#         # unnormalize the image
#         image = train_data[closest_images[i][j]].reshape(c, h, w).transpose(1, 2, 0)
#         image = image * np.array((0.229, 0.224, 0.225)) + np.array(
#             (0.485, 0.456, 0.406)
#         )
#         image = np.clip(image, 0, 1)
#         ax.imshow(image)
#         ax.axis("off")
#         ax.set_title(
#             f"GT Label: {class_idx_to_name[int(train_category[closest_images[i][j]])]}"
#         )

# fig.tight_layout()
# fig.savefig("cifar10_closest_images.png")


# # Visualize the clusters with tsne
# # Reduce dimensionality of the data using t-SNE
# # Plot the clusters using t-SNE with individual colors for each cluster
# from sklearn.manifold import TSNE

# # Reduce the dimensionality of the data using t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_features = tsne.fit_transform(train_data)

# # Create a DataFrame with the reduced features
# # tsne_df = pd.DataFrame(tsne_features, columns=['X', 'Y'])
# # tsne_df['Cluster'] = kmeans.labels_
# # tsne_df['GT Label'] = train_category

# # assign cluster an index based on majority vote of top 100 closest images
# # cluster_indices = []
# # for i in range(10):
# #     distances = np.linalg.norm(train_data - centers[i], axis=1)
# #     closest_indices = np.argsort(distances)[:100]
# #     cluster_indices.append(closest_indices)

# # # Add the cluster labels to the original data
# # df['Cluster'] = clusters

# # # Calculate silhouette score for K=4
# # silhouette_avg = silhouette_score(scaled_features, clusters)

# # # Visualizing the result
# # plt.figure(figsize=(10, 6))
# # sns.scatterplot(x='Fresh', y='Grocery', hue='Cluster', data=df, palette='deep', s=100)
# # plt.title('K-Means Clustering (K=4) - Fresh vs Grocery')
# # plt.xlabel('Fresh Products Spending')
# # plt.ylabel('Grocery Products Spending')
# # plt.grid(True)
# # plt.show()

# # # Show silhouette score for K=4
# # print(f'Silhouette Score for K=4: {silhouette_avg:.4f}')
