import torch
import torch.nn.functional as F
import numpy as np
import time

def predict_anomaly(model, enc2, edge_feat, label, node_map, adj, data_idx, config, device, amplify=True, method='exp'):
    model.eval()

    if data_idx.dtype == bool:
        data_idx = np.where(data_idx)[0]

    total_loss = 0.0
    anomaly_scores = []
    true_labels = label[data_idx]
    batch_size = config.batch_size
    num_batches = int(np.ceil(len(data_idx) / batch_size))

    with torch.no_grad():
        for batch_num in range(num_batches):
            start_time = time.time()
            start = batch_num * batch_size
            end = min(start + batch_size, len(data_idx))
            batch_edges = data_idx[start:end]

            reconstructed, _ = model(batch_edges)

            node_ids_1 = [node_map[adj[edge][0]] for edge in batch_edges]
            node_ids_2 = [node_map[adj[edge][1]] for edge in batch_edges]

            embeds1 = enc2(node_ids_1).to(device)
            embeds2 = enc2(node_ids_2).to(device)

            if embeds1.shape[0] != len(batch_edges):
                embeds1 = embeds1.t()
            if embeds2.shape[0] != len(batch_edges):
                embeds2 = embeds2.t()

            if config.residual:
                edge_feats_batch = edge_feat[batch_edges]
                edge_feats_tensor = torch.FloatTensor(edge_feats_batch).to(device)
                original = torch.cat([embeds1, embeds2, edge_feats_tensor], dim=1)
            else:
                original = torch.cat([embeds1, embeds2], dim=1)

            loss = F.mse_loss(reconstructed, original, reduction='none')
            loss = loss.mean(dim=1)
            anomaly_scores_batch = loss.cpu().numpy()

            if amplify:
                if method == 'exp':
                    min_score = np.min(anomaly_scores_batch)
                    adjusted_scores = anomaly_scores_batch - min_score
                    amplified_batch_scores = np.exp(adjusted_scores)
                elif method == 'square':
                    amplified_batch_scores = np.square(anomaly_scores_batch)
                elif method == 'log':
                    amplified_batch_scores = np.log1p(anomaly_scores_batch)
                else:
                    raise ValueError(f"Phương pháp khuếch đại không xác định: {method}")
            else:
                amplified_batch_scores = anomaly_scores_batch

            anomaly_scores.extend(amplified_batch_scores)
            total_loss += loss.sum().item()

            if (batch_num % 100 == 0):
                print(f'Batch: {batch_num + 1:03d}/{num_batches}, Loss: {loss.mean().item():.4f}, Time: {time.time() - start_time:.4f}s')

    print(f"Average Loss: {total_loss / len(data_idx):.4f}")
    return anomaly_scores, true_labels, data_idx