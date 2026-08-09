import pytest


def test_image_mil_forward_shape():
    torch = pytest.importorskip("torch")
    from orion.models.architectures.image_only import ImageOnlyMILModel

    model = ImageOnlyMILModel(backbone_name="resnet50_scratch", num_classes=3)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 2, 1, 64, 64), slice_mask=torch.ones(1, 2, dtype=torch.bool))
    assert output.shape == (1, 3)
