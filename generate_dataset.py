import os
import numpy as np
import pandas as pd

# Esta función genera un dataset sintético de sensores de servidores con características realistas para estados normales y de fallo
def generate_server_data(
    n_samples: int = 8000,
    random_state: int = 42,
    failure_ratio: float = 0.25
) -> pd.DataFrame:
    np.random.seed(random_state)

    n_failures = int(n_samples * failure_ratio)
    n_normal = n_samples - n_failures

    # Estados normales
    normal_load = np.random.beta(a=2.5, b=3.5, size=n_normal)

    normal_cpu = np.clip(22 + 58 * normal_load + np.random.normal(0, 7, n_normal), 0, 100)
    normal_memory = np.clip(30 + 50 * normal_load + np.random.normal(0, 8, n_normal), 0, 100)
    normal_disk = np.clip(50 + 140 * normal_load + np.random.normal(0, 20, n_normal), 0, None)
    normal_network = np.clip(100 + 380 * normal_load + np.random.normal(0, 50, n_normal), 0, None)

    normal_temp = np.clip(
        34 + 0.30 * normal_cpu + 0.012 * normal_network + np.random.normal(0, 3.5, n_normal),
        20,
        95
    )

    normal_power = np.clip(
        95 + 1.0 * normal_cpu + 0.55 * normal_temp + np.random.normal(0, 10, n_normal),
        50,
        None
    )

    normal_latency = np.clip(
        10 + 0.10 * normal_cpu + 0.08 * normal_memory + 0.015 * normal_network + np.random.normal(0, 5, n_normal),
        1,
        None
    )

    normal_error = np.clip(
        0.3 + 0.010 * normal_cpu + 0.012 * normal_memory + 0.012 * normal_latency + np.random.normal(0, 0.4, n_normal),
        0,
        5
    )
    # Estados de fallo
    failure_load = np.random.beta(a=3.8, b=2.2, size=n_failures)

    failure_cpu = np.clip(38 + 50 * failure_load + np.random.normal(0, 8, n_failures), 0, 100)
    failure_memory = np.clip(45 + 42 * failure_load + np.random.normal(0, 9, n_failures), 0, 100)
    failure_disk = np.clip(90 + 150 * failure_load + np.random.normal(0, 24, n_failures), 0, None)
    failure_network = np.clip(220 + 430 * failure_load + np.random.normal(0, 60, n_failures), 0, None)

    failure_temp = np.clip(
        42 + 0.34 * failure_cpu + 0.015 * failure_network + np.random.normal(0, 4, n_failures),
        25,
        110
    )

    failure_power = np.clip(
        110 + 1.05 * failure_cpu + 0.60 * failure_temp + np.random.normal(0, 12, n_failures),
        60,
        None
    )

    failure_latency = np.clip(
        22 + 0.14 * failure_cpu + 0.10 * failure_memory + 0.022 * failure_network + np.random.normal(0, 7, n_failures),
        1,
        None
    )

    failure_error = np.clip(
        0.9 + 0.014 * failure_cpu + 0.016 * failure_memory + 0.020 * failure_latency + np.random.normal(0, 0.55, n_failures),
        0,
        8
    )
    # Construccion del dataframe
    normal_df = pd.DataFrame({
        "temperature_c": np.round(normal_temp, 2),
        "cpu_usage_pct": np.round(normal_cpu, 2),
        "memory_usage_pct": np.round(normal_memory, 2),
        "disk_io_mb_s": np.round(normal_disk, 2),
        "network_traffic_mb_s": np.round(normal_network, 2),
        "power_consumption_w": np.round(normal_power, 2),
        "latency_ms": np.round(normal_latency, 2),
        "error_rate_pct": np.round(normal_error, 3),
        "failure": 0
    })

    failure_df = pd.DataFrame({
        "temperature_c": np.round(failure_temp, 2),
        "cpu_usage_pct": np.round(failure_cpu, 2),
        "memory_usage_pct": np.round(failure_memory, 2),
        "disk_io_mb_s": np.round(failure_disk, 2),
        "network_traffic_mb_s": np.round(failure_network, 2),
        "power_consumption_w": np.round(failure_power, 2),
        "latency_ms": np.round(failure_latency, 2),
        "error_rate_pct": np.round(failure_error, 3),
        "failure": 1
    })

    df = pd.concat([normal_df, failure_df], axis=0)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_server_data(
        n_samples=8000,
        random_state=42,
        failure_ratio=0.25
    )

    output_path = "server_sensor_data.csv"
    df.to_csv(output_path, index=False)

    print("First rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nFailure distribution:")
    print(df["failure"].value_counts(normalize=True))

    print("\nSaved file:")
    print(os.path.abspath(output_path))