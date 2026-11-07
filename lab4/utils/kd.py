
import torch
import torch.nn.functional as F


def response_kd_loss(z_s, z_t, y, alpha=0.3, beta=0.7,
                     tau=4.0, ignore_index=255):
    # Cross-entropy loss using the ground-truth labels
    hard_loss = F.cross_entropy(
        z_s, y, ignore_index=ignore_index
    )

    # Teacher probabilities with temperature scaling
    teacher_probs = F.softmax(z_t / tau, dim=1)

    # Student log-probabilities with temperature scaling
    student_log_probs = F.log_softmax(z_s / tau, dim=1)

    # Distillation loss for each pixel
    soft_loss = -(
        teacher_probs * student_log_probs
    ).sum(dim=1)

    # Only include valid pixels
    valid = y != ignore_index
    soft_loss = soft_loss[valid].mean()

    # Combine the two losses
    return alpha * hard_loss + beta * soft_loss


def feature_kd_loss(f_s, f_t):
    # Resize student features to match teacher dimensions
    f_s = F.interpolate(
        f_s,
        size=f_t.shape[2:],
        mode="bilinear",
        align_corners=False
    )

    # Calculate cosine similarity across channels
    cosine_sim = F.cosine_similarity(
        f_s, f_t, dim=1
    )

    # Convert similarity into loss
    return (1 - cosine_sim).mean()
