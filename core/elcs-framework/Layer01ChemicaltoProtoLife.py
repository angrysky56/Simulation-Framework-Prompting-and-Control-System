import numpy as np
import pandas as pd
from transformerlab_client.client import TransformerLabClient

# Connect to Transformerlab
client = TransformerLabClient(server_url="http://localhost:8338")

def generate_chemical_system(num_entities=100, time_steps=50):
    """Generate a dataset of chemical components with potential for self-organization"""

    # Create initial chemical entities with random properties
    entities = []
    for i in range(num_entities):
        entity = {
            "id": f"chem_{i}",
            "energy_level": np.random.uniform(0.1, 0.9),
            "bond_capacity": np.random.randint(1, 5),
            "stability": np.random.uniform(0.3, 0.9),
            "reaction_affinity": np.random.uniform(0.1, 0.8),
            "layer": 0,  # Chemical layer
            "neighbors": []
        }
        entities.append(entity)

    # Simulation data through time
    simulation_data = []

    # Run simulation steps
    for step in range(time_steps):
        # Update neighbor relationships based on proximity
        # This is simplified - in a real simulation, we'd use spatial positions
        for i, entity in enumerate(entities):
            entity["neighbors"] = [j for j in range(num_entities) if j != i][:np.random.randint(0, 5)]

        # Check for potential emergent patterns (proto-life formation)
        for i, entity in enumerate(entities):
            # Emergence conditions: high energy, multiple connections, good stability
            if (entity["layer"] == 0 and
                entity["energy_level"] > 0.7 and
                len(entity["neighbors"]) > 3 and
                entity["stability"] > 0.6 and
                entity["reaction_affinity"] > 0.7 and
                np.random.random() < 0.1):  # Random chance of emergence

                # Transition to Layer 1 (Proto-life)
                entity["layer"] = 1
                entity["replication_potential"] = np.random.uniform(0.3, 0.6)
                entity["template_integrity"] = np.random.uniform(0.7, 0.9)

                print(f"Emergence event at step {step}: Entity {entity['id']} transitioned to Layer 1")

        # Store the current state
        for entity in entities:
            record = entity.copy()
            record["time_step"] = step
            simulation_data.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(simulation_data)
    return df

# Generate dataset
simulation_df = generate_chemical_system(num_entities=100, time_steps=50)

# Save as CSV for Transformerlab
simulation_df.to_csv("layer0_to_layer1_simulation.csv", index=False)

# Upload dataset to Transformerlab (if the API supports this)
# We'd need to check the specific API capabilities
try:
    # This is conceptual - actual API call would depend on Transformerlab client implementation
    dataset_id = client.upload_dataset("layer0_to_layer1_simulation.csv", name="Evolutionary_Layer0_to_Layer1")
    print(f"Dataset uploaded with ID: {dataset_id}")
except:
    print("Direct upload not supported - please manually import the CSV in Transformerlab")