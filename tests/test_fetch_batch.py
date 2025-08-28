import torch
from drift import dynamic_mnist_cbp

def test_fetch_batch_matches_per_item():
    """Vectorised batch retrieval mirrors per-item access.

    The training script originally looped over each sampled index, invoking
    ``MNIST.__getitem__`` and ``ToTensor`` repeatedly, which dominated runtime.
    ``fetch_batch`` should return the same binarised images and labels without
    that Python overhead, so this test compares it to the manual loop.
    """
    train, _ = dynamic_mnist_cbp.get_data()
    idxs = torch.randint(len(train), (8,))

    # Manual per-item retrieval replicating the old slow path.
    manual_x, manual_y = [], []
    torch.manual_seed(0)
    for i in idxs.tolist():
        x, y = train[i]
        manual_x.append(torch.bernoulli(x))
        manual_y.append(y)
    manual_x = torch.stack(manual_x)
    manual_y = torch.tensor(manual_y)

    # New vectorised helper under test.
    torch.manual_seed(0)
    x, y = dynamic_mnist_cbp.fetch_batch(train, idxs)

    assert torch.equal(x, manual_x)
    assert torch.equal(y, manual_y)
