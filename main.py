import streamlit as st 
import pandas as pd
from collections import Counter
import numpy as np 

def find_shortest_paths(possible_lengths, start, end, step):
    # Sort the possible lengths in descending order
    possible_lengths.sort(reverse=True)
    
    # Create a dictionary to store the shortest path for each value
    shortest_paths = {round(start + i * step, 1): [] for i in range(int((end - start) / step) + 1)}
    
    # Create a dictionary to store the minimum lengths needed for each value
    min_length_dict = {round(start + i * step, 1): float('inf') for i in range(int((end - start) / step) + 1)}
    min_length_dict[start] = 0

    # Iterate over each possible value from start to end with the given step
    for value in shortest_paths.keys():
        for length in possible_lengths:
            prev_value = round(value - length, 1)
            if prev_value >= start and min_length_dict[prev_value] + 1 < min_length_dict[value]:
                min_length_dict[value] = min_length_dict[prev_value] + 1
                shortest_paths[value] = shortest_paths[prev_value] + [length]
    
    to_cut = set()
    # clean dict:
    for i in np.arange(end, start, -step):
        i = round(i, 1)
        j = round(i - step, 1)
        if min_length_dict[j] == float("inf"):
            to_cut.add(j)
            min_length_dict[j] = min_length_dict[i]
            shortest_paths[j] = shortest_paths[i]

    return shortest_paths, to_cut


def main():
    angle_length: float = 0.3
    possible_lengths: list[float] = [0.3, 0.6, 1, 1.2, 1.3, 1.5, 2, 3]

    num_sides: int = int(st.number_input("How many sides: ", step=1))
    sides, angles, res = dict(), dict(), dict()

    for i in range(0, num_sides):
        sides[i] = st.number_input(f"What is the length of the side {i + 1}: ", value=None, key=f"side_len_{i}")
        num_angles = 1 if i in [0, num_sides - 1] else 2
        if num_sides == 1:
            num_angles = 0
        if num_sides == 4:
            num_angles = 2
        angles[i] = num_angles

    # only_smaller: bool = True if st.checkbox(f'Get only smaller possibilities if, can not have exact distance') else False

    if st.button("Calculate : "):
        max_size = max(sides.values()) + 1
        paths_dict, to_cuts = find_shortest_paths(start=0, end=max_size, possible_lengths=possible_lengths, step=0.1)
        for k, v in sides.items():
            res[k] = round(v - angles[k] * angle_length, 1), paths_dict.get(round(v - angles[k] * angle_length, 1))

        df_dict = {"side":[], "num_angles":[], "full_distance": [], 
                   "distance_after_angle":[], "distance_got": [], 
                   "leds": [], "need_to_cut": []}
        for k, (dist, v) in res.items():
            distance_after_angle = round(sides[k] - angles[k] * angle_length, 2)
            df_dict["side"].append(k+1)
            df_dict["num_angles"].append(angles.get(k))
            df_dict["full_distance"].append(sides.get(k))
            df_dict["distance_after_angle"].append(distance_after_angle)
            df_dict["distance_got"].append(round(sum(v), 1))
            df_dict["leds"].append(dict(Counter(v)))
            df_dict["need_to_cut"].append(dist in to_cuts)
        st.table(pd.DataFrame(df_dict))

        # canvas = np.zeros((max_size + 10, max_size + 10))
        # get_color = {length: i * 10 + 10 for i, length in enumerate(possible_lengths)}
        # get_color['angle'] = 10

        # start_point = (10, 10)
        # thickness = 1

        # for k, angle in angles.items():
        #     if k == 0 and angle > 1:
                
        #         canvas = cv2.line(canvas, start_point, end_point, get_color("angle"), thickness) 
        #         image = cv2.line(image, start_point, end_point, color, thickness) 






if __name__ == "__main__":
    main()
