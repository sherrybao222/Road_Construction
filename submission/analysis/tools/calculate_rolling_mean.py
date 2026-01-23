def calculate_rolling_mean(group):
    return group['numCities'].expanding().mean().shift(1)