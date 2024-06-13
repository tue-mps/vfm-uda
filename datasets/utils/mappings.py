import torchvision


def get_cityscapes_mapping():
    cityscapes_mapping = {
        class_.id: class_.train_id for class_ in torchvision.datasets.Cityscapes.classes
    }
    return cityscapes_mapping


def get_mapillary_mapping():
    mapillary_mapping = {
        13: 0,
        24: 0,
        41: 0,
        2: 1,
        15: 1,
        17: 2,
        6: 3,
        3: 4,
        45: 5,
        47: 5,
        48: 6,
        50: 7,
        30: 8,
        29: 9,
        27: 10,
        19: 11,
        20: 12,
        21: 12,
        22: 12,
        55: 13,
        61: 14,
        54: 15,
        58: 16,
        57: 17,
        52: 18,
    }
    return mapillary_mapping


def get_synthia_mapping():
    synthia_mapping = {
        0: 255,  # Void
        1: 10,  # Sky
        2: 2,  # Building
        3: 0,  # Road
        4: 1,  # Sidewalk
        5: 4,  # Fence
        6: 8,  # Vegetation
        7: 5,  # Pole
        8: 13,  # Car
        9: 7,  # Traffic Sign
        10: 11,  # Pedestrian
        11: 18,  # Bicycle
        12: 17,  # Motorcycle
        13: 255,  # Parking-slot
        14: 255,  # Road-work
        15: 6,  # Traffic Light
        16: 9,  # Terrain
        17: 12,  # Rider
        18: 14,  # Truck
        19: 15,  # Bus
        20: 16,  # Train
        21: 3,  # Wall
        22: 255  # Lanemarking
    }
    return synthia_mapping


def get_label2name():
    return {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic_light",
        7: " traffic_sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle"
    }
