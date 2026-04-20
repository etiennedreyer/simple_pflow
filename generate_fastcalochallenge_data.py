import h5py
import argparse
import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax_calo_flash.calorimeter import CaloCylinder

def read_calochallenge_data(file_path, entry_start, entry_stop=None):

    if entry_stop is None or entry_stop < 0:
        with h5py.File(file_path, "r") as f:
            entry_stop = len(f["incident_energies"])

    n_events = entry_stop - entry_start
    print(f"Reading {n_events} entries from {file_path}...")
    with h5py.File(file_path, "r") as f:
        data = {
            "incident_energies": f["incident_energies"][entry_start:entry_stop],
            "showers": f["showers"][entry_start:entry_stop],
        }

    return data

def reshape_data_for_calochallenge(calo_dict):

    incident_energies = calo_dict["particle_e"]
    showers = calo_dict["cell_e"]
    showers = jnp.permute_dims(showers, (0, 3, 2, 1))  # (N_events, N_z, N_r, N_phi)
    showers = jnp.reshape(showers, (showers.shape[0], -1))  # (N_events, N_cells)

    return incident_energies, showers

def parse_args():
    parser = argparse.ArgumentParser(description="Generate FastCaloChallenge data using JAX CaloFlash")
    parser.add_argument("--config", "-c", type=str, help="Path to config YAML file for data generation")
    parser.add_argument("--input_path", "-i", type=str, help="Path to input HDF5 file")
    parser.add_argument("--output_path", "-o", type=str, default="calo_flash_data.hdf5", help="Path to output HDF5 file")
    parser.add_argument("--batch_size", "-bs", type=int, default=64, help="Number of events to generate per batch")
    parser.add_argument("--n_events", "-n", type=int, default=-1, help="Number of events to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    calo = CaloCylinder(args.config)
    input_data = read_calochallenge_data(args.input_path, 0, args.n_events)
    incident_energies = jnp.array(input_data["incident_energies"])

    n_events = len(incident_energies)
    if args.n_events > 0:
        n_events = min(n_events, args.n_events)
        incident_energies = incident_energies[:n_events]
    if args.batch_size > n_events:
        args.batch_size = n_events

    ### Prepare output file
    out_file = h5py.File(args.output_path, "w")
    out_file.create_dataset("incident_energies", shape=(n_events, 1), dtype="float32")
    out_file.create_dataset("showers", shape=(n_events, calo.N_cells_tot), dtype="float32")

    for i in tqdm(range(0, len(incident_energies), args.batch_size)):
        batch_energies = incident_energies[i:i+args.batch_size]
        batch_dict = calo.simulate(batch_energies, return_hits=False, return_truth=False)
        batch_incident_energies, batch_showers = reshape_data_for_calochallenge(batch_dict)
        out_file["incident_energies"][i:i+args.batch_size] = batch_incident_energies
        out_file["showers"][i:i+args.batch_size] = batch_showers

    out_file.close()
    print(f"Output written to {args.output_path}")

if __name__ == "__main__":
    main()