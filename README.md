# NiNformer

NiNformer: A Network in Network Transformer with Token Mixing Generated Gating Function

Abdullah Nazhat Abdullah, Tarkan Aydin

# Abstract

The attention mechanism is the main component of the transformer architecture, and since its introduction, it has led to significant advancements in deep learning that span many domains and multiple tasks. The attention mechanism was utilized in computer vision as the Vision Transformer ViT, and its usage has expanded into many tasks in the vision domain, such as classification, segmentation, object detection, and image generation. While this mechanism is very expressive and capable, it comes with the drawback of being computationally expensive and requiring datasets of considerable size for effective optimization. To address these shortcomings, many designs have been proposed in the literature to reduce the computational burden and alleviate the data size requirements. Examples of such attempts in the vision domain are the MLP-Mixer, the Conv-Mixer, the Perciver-IO, and many more. This paper introduces a new computational block as an alternative to the standard ViT block that reduces the compute burdens by replacing the normal attention layers with a Network in Network structure that enhances the static approach of the MLP-Mixer with a dynamic system of learning an element-wise gating function by a token mixing process. Extensive experimentation shows that the proposed design provides better performance than the baseline architectures on multiple datasets applied in the image classification task of the vision domain.

Paper Link : https://arxiv.org/abs/2403.02411v3
