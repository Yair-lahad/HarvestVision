class FruitCounter:
    def __init__(self):
        self.unique_ids = set()

    def update(self, tracked_objects):
        for track_id, _ in tracked_objects:
            self.unique_ids.add(track_id)

    def total_count(self):
        return len(self.unique_ids)

    def reset(self):
        self.unique_ids.clear()
