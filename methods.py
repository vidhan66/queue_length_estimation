import json
from pathlib import Path

import vehicle_queue_simulation 
def process_arrival(vehicle_type, length, width, lateral_clearance, longitudinal_gap, 
                    timestamp):
    
    effective_width = width + lateral_clearance(vehicle_type)

    vehicle_queue_simulation.add_vehicle(
        length=length,
        effective_width=effective_width,
        gap=longitudinal_gap(vehicle_type),
        timestamp=timestamp,
    )


def get_vehicle_type(vehicle_data):
    """Add a dimension-based vehicle_type to each vehicle record.

    The thresholds are in meters. Car and SUV dimensions overlap, so both are
    assigned to Car; Bus is distinguished from Truck using the dimensions in
    the supplied vehicle set.
    """
    for vehicle in vehicle_data:
        length = float(vehicle["length_m"])
        width = float(vehicle["width_m"])

        if length <= 0 or width <= 0:
            raise ValueError("Vehicle length and width must be positive.")
        if length <= 1.3 and width <= 0.75:
            vehicle_type = "Pedestrian"
        elif length <= 2.0 and width <= 0.8:
            vehicle_type = "Two-wheeler"
        elif length <= 5.0 and width <= 2.2:
            vehicle_type = "Car"
        elif length <= 7.0 and width <= 2.5:
            vehicle_type = "LCV"
        elif 10 <= length <= 14.0 and width >= 2.4:
            vehicle_type = "Bus"
        elif 14 <= length <= 20.0 and width >= 2.4:
            vehicle_type = "Truck"
        else:
            vehicle_type = "TramTrain"

        vehicle["vehicle_type"] = vehicle_type

    return vehicle_data

def get_spacing(vehicle_data):
    """Add configured spacing values to each vehicle record."""
    spacing_file = Path(__file__).with_name("spacing_dict.json")
    with spacing_file.open(encoding="utf-8") as file:
        spacing = json.load(file)["per_vehicle_spacing"]

    for vehicle in vehicle_data:
        vehicle_type = vehicle["vehicle_type"]
        if vehicle_type not in spacing:
            raise ValueError(
                f"No spacing values configured for vehicle type "
                f"{vehicle_type!r} in {spacing_file.name}."
            )
        vehicle.update(spacing[vehicle_type])

    return vehicle_data