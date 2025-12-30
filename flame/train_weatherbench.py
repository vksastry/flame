
from data_weatherbench import WeatherBenchTemporalDataset
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    device = 'cpu'
def main():
    ds = WeatherBenchTemporalDataset(
        path="/path/to/weatherbench.zarr",
        variables=["2m_temperature"],
        in_len=24,
        out_len=24,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

    model = TemporalDeltaNet(d_model=256, n_layers=4, ...)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        for batch in loader:
            x = batch["inputs"].to(device)   # (B, T_in, C, H, W)
            y = batch["targets"].to(device)

            # your flattening / embedding goes here
            # e.g., x -> (B, T_in, S, C) -> (B, T_in, d_model)

            pred = model(x_seq)
            loss = loss_fn(pred, y_seq)

            opt.zero_grad()
            loss.backward()
            opt.step()

