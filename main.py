import streamlit as st 
import pandas as pd
from collections import Counter
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np

import plotly.express as px
import plotly.graph_objects as go


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

get_color = {
    0.3: (100, 255, 0),
    0.6: (255, 0, 0), 
    1 :(125, 255, 255),
    1.2 : (0, 255, 156), 
    1.3 :(255, 0, 255),
    1.5: (125, 125, 125), 
    2: (0, 0, 255),
    3: (255, 120, 120),
    "angle": (160, 39, 46)
}


def add_points_to_canvas(canvas: np.ndarray, point: np.ndarray, insert_point: int, radius: int = 5) -> tuple[np.ndarray, int]:
    # canvas = cv2.circle(canvas, (point[0], point[1]), radius, (1, 1, 1), -1) 
    # canvas = cv2.putText(canvas, str(insert_point), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX ,  
    #                 1, (255, 255, 255), 2, cv2.LINE_AA) 
    canvas[point[0]][point[1]] = (-1, -1, -1)
    return canvas, insert_point + 1


def main(debug: bool = False):
    st.markdown("""
        <style>
        input {
        unicode-bidi:bidi-override;
        direction: RTL;
        }
        </style>
            """, unsafe_allow_html=True)

    angle_length: float = 0.3
    possible_lengths: list[float] = [0.2, 0.3, 0.6, 1, 1.2, 1.3, 1.5, 2, 3]

    num_sides: int = int(st.number_input("כמה צדדים: ", step=1))
    sides, angles, res = dict(), dict(), dict()
    if debug:
        num_sides = 2
        sides[0] = 4.5
        sides[1] = 3.5
    for i in range(0, num_sides):
        if not debug:
            sides[i] = st.number_input(f"מה אורך של הצד ה{i + 1 } : ", value=None, key=f"side_len_{i}")
        num_angles = 1 if i in [0, num_sides - 1] else 2
        if num_sides == 1:
            num_angles = 0
        if num_sides == 4:
            num_angles = 2
        angles[i] = num_angles
    
    profil_color = st.selectbox(label="צבע פרופיל",
                                options=["לבן", " כסוף", "שחור", "אחר"]
                                )
    light_intesity = st.selectbox(label="גוון אור",
                                options=["3000K", "4000k", "6000k"])
    instalation_way = st.selectbox(label="אופן התקנה", 
                                   options=["צמוד", "שקוע", "תלוי + רוזטה", "שקוע טרינלס"])
    # only_smaller: bool = True if st.checkbox(f'Get only smaller possibilities if, can not have exact distance') else False

    if st.button("לחשב אפשרויות :"):
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

    directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
    if st.button("הראה עיור : ") or debug:
        max_size = max(sides.values()) + 1
        paths_dict, to_cuts = find_shortest_paths(start=0, end=max_size, possible_lengths=possible_lengths, step=0.1)
        for k, v in sides.items():
            res[k] = round(v - angles[k] * angle_length, 1), paths_dict.get(round(v - angles[k] * angle_length, 1))
        const_size_multiplier_for_int = 100
        canvas = np.zeros((int(max_size * const_size_multiplier_for_int) * 2, 
                           int(max_size * const_size_multiplier_for_int) * 2, 3))
        cur_point = np.array([int(max_size * const_size_multiplier_for_int) // 2, 5])
        thickness = 2
        on_x = True 
        done_angles = 0
        insert_points = []
        for idx, (k, (dist, v)) in enumerate(res.items()):
            num_angles = angles.get(k) - done_angles
            for led_size, num_of in dict(Counter(v)).items():
                for _ in range(num_of):
                    adding = directions[idx] * int(led_size * const_size_multiplier_for_int)
                    canvas = cv2.line(canvas, cur_point, cur_point + adding, get_color[led_size], thickness)
                    canvas, insert_point = add_points_to_canvas(canvas=canvas, point=cur_point, insert_point=0)
                    cur_point = cur_point + adding
                    canvas, insert_point = add_points_to_canvas(canvas=canvas, point=cur_point, insert_point=0)
                
            if num_angles:
                adding = directions[idx] * int(angle_length * const_size_multiplier_for_int)
                second_part_angle = directions[(idx + 1) % len(directions)] * int(angle_length * const_size_multiplier_for_int)
                canvas = cv2.line(canvas, cur_point, cur_point + adding, get_color["angle"], thickness) 
                canvas, insert_point = add_points_to_canvas(canvas=canvas, point=cur_point, insert_point=insert_point) 
                canvas, insert_point = add_points_to_canvas(canvas=canvas, point=cur_point + adding, insert_point=insert_point) 

                canvas = cv2.line(canvas, cur_point + adding, cur_point + adding + second_part_angle, get_color["angle"], thickness) 
                cur_point = cur_point + adding + second_part_angle
                canvas, insert_point = add_points_to_canvas(canvas=canvas, point=cur_point, insert_point=insert_point)
                insert_points.append(cur_point)
                done_angles = 1

            on_x = False
        xs = np.argwhere(canvas > 0)[:, 0]
        ys = np.argwhere(canvas > 0)[:, 1]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        new_canvas_buffer = 50
        new_canvas = np.zeros(((xmax - xmin) + new_canvas_buffer, (ymax - ymin) + new_canvas_buffer , 3), dtype=np.float16)
        new_canvas[new_canvas_buffer //2: -new_canvas_buffer //2, new_canvas_buffer // 2: -new_canvas_buffer // 2, :] = canvas[xmin: xmax, ymin: ymax, :]

        fig, ax = plt.subplots(1)
        handles = [
            Rectangle((0,0),1,1, color = tuple((v/255 for v in c))) for c in get_color.values()
        ]
        labels = [k + " 0.3 | 0.3" if isinstance(k, str) else str(k) for k in get_color.keys()]
        
        insert_points = np.argwhere(canvas == -1)[:, :2]
        insert_points = list(set([((x - xmin) + new_canvas_buffer //2, (y - ymin) + new_canvas_buffer // 2) for y, x in insert_points]))
        fig_image = px.imshow(new_canvas)
        height, width, _ = new_canvas.shape
        
        for _idx, _inner_point in enumerate(insert_points):
              
            fig_image.add_trace(go.Scatter(x=[_inner_point[1]], y=[_inner_point[0]], 
                                           marker=dict(color='white', size=4), text=str(_idx), name=f"Connection point: {_idx}"))

        if num_sides > 1:
            canvas_flip = c2.flip(canvas, 1)
            insert_points = np.argwhere(canvas_flip == -1)[:, :2]
            insert_points = list(set([((x - xmin) + new_canvas_buffer //2, (y - ymin) + new_canvas_buffer // 2) for y, x in insert_points]))
            fig_image_two = px.imshow(cv2.flip(new_canvas, 1)
            height, width, _ = new_canvas.shape
            for _idx, _inner_point in enumerate(insert_points):
            
                fig_image_two.add_trace(go.Scatter(x=[_inner_point[1]], y=[_inner_point[0]], 
                                           marker=dict(color='white', size=4), text=str(_idx), name=f"Connection point: {_idx}"))

        st.plotly_chart(fig_image, theme=None, use_container_width=True)
        if num_sides >1:
            st.plotly_chart(fig_image_two, theme=None, use_container_width=True
                           )
        ax.legend(handles,labels, mode='expand', ncol=3)
        ax.axis('off')
        st.pyplot(fig)


if __name__ == "__main__":
    main(False)
