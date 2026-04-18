import numpy as np

# Parameters from notebook
yearly_reproduction_rate = 1.5
yearly_sunlight_hours = 365 * 24  # 8760
step_size = 1/60  # hours per step

# Calculate reproduction rate per step
reproduction_rate = (yearly_reproduction_rate / yearly_sunlight_hours) * step_size
print(f"Reproduction rate per step: {reproduction_rate}")

# Steps per year
steps_per_year = yearly_sunlight_hours / step_size
print(f"Steps per year: {steps_per_year}")

# Total probability per year
total_prob = reproduction_rate * steps_per_year
print(f"Total probability per year: {total_prob}")

# Simulation parameters
max_steps = 180 * 24 * 60  # 180 days
init_deer = 500
min_breeding_age = 2

# Age increase per step
age_increase_per_step = 1 / yearly_sunlight_hours
print(f"Age increase per step: {age_increase_per_step}")

# Steps to reach breeding age
steps_to_breed = min_breeding_age / age_increase_per_step
print(f"Steps to reach breeding age: {steps_to_breed}")

# Days to reach breeding age
days_to_breed = steps_to_breed / (24 * 60)
print(f"Days to reach breeding age: {days_to_breed}")

# Breeding steps
breeding_steps = max_steps - steps_to_breed
print(f"Breeding steps: {breeding_steps}")

# Assuming half are female
females = init_deer / 2
print(f"Females: {females}")

# Expected births
expected_births = females * reproduction_rate * breeding_steps
print(f"Expected births: {expected_births}")

# Per run
print(f"Expected births per run: {expected_births}")