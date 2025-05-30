from vae import create_vae_model, create_conditiaonl_vae_model
import torch

if __name__ == '__main__':
    vae = create_vae_model(
        input_feature_dim=1,
        sequence_len=100,
        latent_feature_dim=4,
        scale_factor=4,
        backbone_type='transformer'
    )

    cvae = create_conditiaonl_vae_model(
        input_feature_dim=1,
        sequence_len=100,
        latent_feature_dim=4,
        conditional_dim=1,
        scale_factor=4,
        backbone_type='transformer'
    )

    cvae_cross = create_conditiaonl_vae_model(
        input_feature_dim=1,
        sequence_len=100,
        latent_feature_dim=4,
        conditional_dim=1,
        scale_factor=4,
        backbone_type='transformer',
        use_crossattention=True,
    )


    input = torch.randn(64, 100, 1)
    condition = torch.randn(64, 4, 1)
    x_reconstructed, mean, log_var = cvae_cross(input, condition)

    print(x_reconstructed.shape)
    print(mean.shape)