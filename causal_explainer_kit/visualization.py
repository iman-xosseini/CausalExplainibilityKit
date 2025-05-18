# causal_explainer_kit/visualization.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# Define global colors for effects for consistency in edges and legend
POSITIVE_EFFECT_COLOR = "#28a745"  # A clear green
NEGATIVE_EFFECT_COLOR = "#dc3545"  # A clear red
NEUTRAL_EFFECT_COLOR = "#6c757d"   # A neutral grey (for effects near zero if shown)

def display_causal_graph_customized(
    causal_graph,
    target_node_name="Target",
    feature_importance_dict=None, # This is for node sizing, obtained from CausalExplainer.predict feature importances
    edge_threshold=0.0,
    use_edge_threshold=False,
    draw_node_labels=True,
    draw_edge_labels=True,
    node_font_size=9,
    edge_font_size_ratio=0.8,
    node_base_size=1800, # Increased base size slightly
    target_node_scale_factor=1.2,
    figure_size=(13, 9),
    spring_layout_k=0.9, # Adjusted for potentially better spacing
    spring_layout_iterations=80, # Increased iterations
    seed=42,
    edge_label_display_threshold=0.01,
    node_label_y_offset_factor=0.055, # Slightly more offset
    plot_name="random_plot.png"
):
    """
    Displays a causal graph with specified customizations.
    (Docstring from your provided code, slightly adapted)
    """

    # --- 1. Input Validation and Graph Copy ---
    if not isinstance(causal_graph, (nx.DiGraph, nx.Graph)):
        raise TypeError("causal_graph must be a NetworkX DiGraph or Graph.")

    G = causal_graph.copy()

    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            print(f"Warning: Edge ({u}, {v}) missing 'weight' attribute. Defaulting to 0.0.")
            data['weight'] = 0.0

    # --- 2. Identify Target Node and Remove Self-Loops ---
    actual_target_node_name = None
    # Prioritize exact match for target_node_name
    if target_node_name in G:
        actual_target_node_name = target_node_name
    else: # Heuristic if exact name not found but a similar one exists
        for node in G.nodes():
            if isinstance(node, str) and target_node_name.lower() in node.lower():
                actual_target_node_name = node
                print(f"Warning: Exact target node '{target_node_name}' not found. Using '{node}' based on name match.")
                break
    
    if actual_target_node_name and G.has_node(actual_target_node_name):
        if G.has_edge(actual_target_node_name, actual_target_node_name):
            G.remove_edge(actual_target_node_name, actual_target_node_name)
            # print(f"Removed self-loop on target node '{actual_target_node_name}'.") # Kept it commented as per your code
    elif target_node_name: # Only print warning if a target_node_name was given but not found
        print(f"Warning: Specified target node '{target_node_name}' not found in graph. "
              "Cannot remove self-loops or style target node distinctly.")

    # --- 3. Edge Filtering based on Threshold ---
    if use_edge_threshold:
        edges_to_remove = [
            (u, v) for u, v, data in G.edges(data=True)
            if abs(data.get('weight', 0.0)) < edge_threshold
        ]
        G.remove_edges_from(edges_to_remove)

    # --- 4. Handle Empty Graph ---
    if not G.nodes():
        # Avoid trying to plot an empty figure if G has no nodes from the start or after filtering
        print("Graph has no nodes to display.")
        # Optionally, display a blank plot with a message
        plt.figure(figsize=figure_size)
        plt.text(0.5, 0.5, "No data to display (graph is empty).", ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.show()
        return

    # --- 5. Node Attributes (Size, Color, Label) ---
    node_attrs = {}
    for node in G.nodes():
        is_target = (str(node) == str(actual_target_node_name)) if actual_target_node_name else False
        
        # Get importance for sizing
        importance = 0.5  # Default base importance
        if feature_importance_dict and node in feature_importance_dict:
            importance = feature_importance_dict[node]
        elif is_target: # If target, and not in feature_importance_dict, give it max importance for sizing
            importance = 1.0

        # Scale size by importance. Max importance (1.0) gives 1.3x base_size. Min (0.0) gives 1.0x base_size.
        # Adjusted importance scaling for more visual difference
        current_node_base_size = node_base_size * (1 + importance * 0.5)
        if is_target:
            current_node_base_size *= target_node_scale_factor

        node_attrs[node] = {
            "size": current_node_base_size,
            "color": "#FF6347" if is_target else "#1E90FF",
            "label": str(node),
            "is_target": is_target
        }
    node_sizes_list = [node_attrs[node]["size"] for node in G.nodes()]
    node_colors_list = [node_attrs[node]["color"] for node in G.nodes()]
    node_labels_dict = {node: node_attrs[node]["label"] for node in G.nodes()}

    # --- 6. Edge Effects and Styling Data ---
    edge_effects = {}
    edge_styles_data = [] 

    for u, v, data in G.edges(data=True):
        effect = data.get('weight', 0.0)
        edge_effects[(u, v)] = effect

        if effect > 0.001: edge_color = POSITIVE_EFFECT_COLOR
        elif effect < -0.001: edge_color = NEGATIVE_EFFECT_COLOR
        else: edge_color = NEUTRAL_EFFECT_COLOR

        base_width = min(4.0, max(0.5, abs(effect) * 3.0))
        arrow_s = 15
        alpha_val = 0.75

        is_u_target = node_attrs.get(u, {}).get("is_target", False)
        is_v_target = node_attrs.get(v, {}).get("is_target", False)
        if is_u_target or is_v_target: # Emphasize edges connected to target
            base_width = min(5.0, max(1.0, abs(effect) * 4.0))
            arrow_s = 20
            alpha_val = 0.9
        edge_styles_data.append(((u,v), base_width, edge_color, arrow_s, alpha_val))

    # --- 7. Layout ---
    if not G.nodes(): pos = {} # Should be caught earlier
    elif len(G.nodes()) == 1: pos = {list(G.nodes())[0]: np.array([0.5, 0.5])}
    else: pos = nx.spring_layout(G, k=spring_layout_k, iterations=spring_layout_iterations, seed=seed, dim=2)

    # --- 8. Drawing Nodes ---
    plt.figure(figsize=figure_size, facecolor="white")
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes_list, node_color=node_colors_list,
        alpha=0.9, edgecolors="gray", linewidths=1.0
    )

    # --- 9. Drawing Edges ---
    for (u,v), width, color, arr_size, alpha_val in edge_styles_data:
        # Calculate margins based on actual node sizes at positions
        # This is an approximation for node radius in plot coordinates
        # These margins help arrows not to be drawn right from/to the center of large nodes
        # This part is experimental and might need tuning based on plot scale and node sizes
        # For simplicity, using fixed small value or scaled from base size for margin
        source_node_size_attr = node_attrs.get(u, {"size": node_base_size})["size"]
        target_node_size_attr = node_attrs.get(v, {"size": node_base_size})["size"]
        # Heuristic for margin, may need adjustment if node sizes vary extremely
        margin_scale_factor = 30000 # Adjust this based on typical node_base_size and figure size
        
        source_margin = np.sqrt(source_node_size_attr) / np.sqrt(margin_scale_factor)
        target_margin = np.sqrt(target_node_size_attr) / np.sqrt(margin_scale_factor)


        nx.draw_networkx_edges(
            G, pos, edgelist=[(u,v)], width=width, edge_color=color,
            arrowsize=arr_size, arrowstyle="-|>", connectionstyle="arc3,rad=0.1",
            alpha=alpha_val,
            # The min_source_margin and min_target_margin are relative to the data coordinates.
            # Their effect depends on the scale of `pos` and node sizes.
            # Using a more direct scaling:
            # min_source_margin = 0.01 + 0.00001 * source_node_size_attr, # Experiment
            # min_target_margin = 0.01 + 0.00001 * target_node_size_attr  # Experiment
        )
    
    # --- 10. Drawing Node Labels ---
    if draw_node_labels:
        label_pos_adjusted = {
            node: (coords[0], coords[1] + node_label_y_offset_factor * (np.sqrt(node_attrs[node]["size"] / (np.pi * node_base_size))))
            for node, coords in pos.items() if node in node_attrs # Ensure node exists
        }
        nx.draw_networkx_labels(
            G, label_pos_adjusted, labels=node_labels_dict, font_size=node_font_size,
            font_family="sans-serif", font_weight="normal", font_color="#333333"
        )

    # --- 11. Drawing Edge Labels ---
    if draw_edge_labels and G.edges():
        final_edge_labels = {}
        for u, v in G.edges():
            effect = edge_effects.get((u, v), 0.0)
            is_u_target = node_attrs.get(u,{}).get("is_target",False)
            is_v_target = node_attrs.get(v,{}).get("is_target",False)
            is_target_edge = is_u_target or is_v_target

            if abs(effect) >= edge_label_display_threshold or \
               (is_target_edge and abs(effect) > 0.001):
                final_edge_labels[(u, v)] = f"{effect:.2f}"

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=final_edge_labels,
            font_size=max(7, int(node_font_size * edge_font_size_ratio)),
            font_color="#222222",
            bbox=dict(facecolor="white", alpha=0.75, boxstyle="round,pad=0.2", lw=0.3, edgecolor='lightgrey'),
            label_pos=0.5, rotate=False
        )

    # --- 12. Plot Aesthetics ---
    plt.title("Causal Relationships Graph", fontsize=16, fontweight="bold", pad=20, color="#111111")
    plt.axis("off")
    
    # --- 13. Simplified Legend ---
    legend_elements = [
        plt.Line2D([0], [0], color=POSITIVE_EFFECT_COLOR, lw=3, label="Positive Effect"),
        plt.Line2D([0], [0], color=NEGATIVE_EFFECT_COLOR, lw=3, label="Negative Effect")
    ]
    # Check if any neutral edges were actually drawn to decide if to add to legend
    if any(style_data[2] == NEUTRAL_EFFECT_COLOR for _, _, style_data, _, _ in edge_styles_data if style_data):
         legend_elements.append(plt.Line2D([0], [0], color=NEUTRAL_EFFECT_COLOR, lw=3, label="Neutral/Zero Effect"))

    plt.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_elements), fontsize=max(9, int(node_font_size)), frameon=False
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Apply after legend to consider its space

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../image"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, plot_name)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"Plot {plot_name} saved to: {output_path}")

    plt.show()
    




# def display_causal_graph_customized(
#     causal_graph,
#     target_node_name="Target",
#     feature_importance_dict=None, # This is for node sizing, obtained from CausalExplainer.predict feature importances
#     edge_threshold=0.0,
#     use_edge_threshold=False,
#     draw_node_labels=True,
#     draw_edge_labels=True,
#     node_font_size=9,
#     edge_font_size_ratio=0.8,
#     node_base_size=1800, # Increased base size slightly
#     target_node_scale_factor=1.2,
#     figure_size=(13, 9),
#     spring_layout_k=0.9, # Adjusted for potentially better spacing
#     spring_layout_iterations=80, # Increased iterations
#     seed=42,
#     edge_label_display_threshold=0.01,
#     node_label_y_offset_factor=0.055 # Slightly more offset
# ):
#     """
#     Displays a causal graph with specified customizations.
#     (Docstring from your provided code, slightly adapted)
#     """

#     # --- 1. Input Validation and Graph Copy ---
#     if not isinstance(causal_graph, (nx.DiGraph, nx.Graph)):
#         raise TypeError("causal_graph must be a NetworkX DiGraph or Graph.")

#     G = causal_graph.copy()

#     for u, v, data in G.edges(data=True):
#         if 'weight' not in data:
#             print(f"Warning: Edge ({u}, {v}) missing 'weight' attribute. Defaulting to 0.0.")
#             data['weight'] = 0.0

#     # --- 2. Identify Target Node and Remove Self-Loops ---
#     actual_target_node_name = None
#     # Prioritize exact match for target_node_name
#     if target_node_name in G:
#         actual_target_node_name = target_node_name
#     else: # Heuristic if exact name not found but a similar one exists
#         for node in G.nodes():
#             if isinstance(node, str) and target_node_name.lower() in node.lower():
#                 actual_target_node_name = node
#                 print(f"Warning: Exact target node '{target_node_name}' not found. Using '{node}' based on name match.")
#                 break
    
#     if actual_target_node_name and G.has_node(actual_target_node_name):
#         if G.has_edge(actual_target_node_name, actual_target_node_name):
#             G.remove_edge(actual_target_node_name, actual_target_node_name)
#             # print(f"Removed self-loop on target node '{actual_target_node_name}'.") # Kept it commented as per your code
#     elif target_node_name: # Only print warning if a target_node_name was given but not found
#         print(f"Warning: Specified target node '{target_node_name}' not found in graph. "
#               "Cannot remove self-loops or style target node distinctly.")

#     # --- 3. Edge Filtering based on Threshold ---
#     if use_edge_threshold:
#         edges_to_remove = [
#             (u, v) for u, v, data in G.edges(data=True)
#             if abs(data.get('weight', 0.0)) < edge_threshold
#         ]
#         G.remove_edges_from(edges_to_remove)

#     # --- 4. Handle Empty Graph ---
#     if not G.nodes():
#         # Avoid trying to plot an empty figure if G has no nodes from the start or after filtering
#         print("Graph has no nodes to display.")
#         # Optionally, display a blank plot with a message
#         plt.figure(figsize=figure_size)
#         plt.text(0.5, 0.5, "No data to display (graph is empty).", ha='center', va='center', fontsize=16)
#         plt.axis('off')
#         plt.show()
#         return

#     # --- 5. Node Attributes (Size, Color, Label) ---
#     node_attrs = {}
#     for node in G.nodes():
#         is_target = (str(node) == str(actual_target_node_name)) if actual_target_node_name else False
        
#         # Get importance for sizing
#         importance = 0.5  # Default base importance
#         if feature_importance_dict and node in feature_importance_dict:
#             importance = feature_importance_dict[node]
#         elif is_target: # If target, and not in feature_importance_dict, give it max importance for sizing
#             importance = 1.0

#         # Scale size by importance. Max importance (1.0) gives 1.3x base_size. Min (0.0) gives 1.0x base_size.
#         # Adjusted importance scaling for more visual difference
#         current_node_base_size = node_base_size * (1 + importance * 0.5)
#         if is_target:
#             current_node_base_size *= target_node_scale_factor

#         node_attrs[node] = {
#             "size": current_node_base_size,
#             "color": "#FF6347" if is_target else "#1E90FF",
#             "label": str(node),
#             "is_target": is_target
#         }
#     node_sizes_list = [node_attrs[node]["size"] for node in G.nodes()]
#     node_colors_list = [node_attrs[node]["color"] for node in G.nodes()]
#     node_labels_dict = {node: node_attrs[node]["label"] for node in G.nodes()}

#     # --- 6. Edge Effects and Styling Data ---
#     edge_effects = {}
#     edge_styles_data = [] 

#     for u, v, data in G.edges(data=True):
#         effect = data.get('weight', 0.0)
#         edge_effects[(u, v)] = effect

#         if effect > 0.001: edge_color = POSITIVE_EFFECT_COLOR
#         elif effect < -0.001: edge_color = NEGATIVE_EFFECT_COLOR
#         else: edge_color = NEUTRAL_EFFECT_COLOR

#         base_width = min(4.0, max(0.5, abs(effect) * 3.0))
#         arrow_s = 15
#         alpha_val = 0.75

#         is_u_target = node_attrs.get(u, {}).get("is_target", False)
#         is_v_target = node_attrs.get(v, {}).get("is_target", False)
#         if is_u_target or is_v_target: # Emphasize edges connected to target
#             base_width = min(5.0, max(1.0, abs(effect) * 4.0))
#             arrow_s = 20
#             alpha_val = 0.9
#         edge_styles_data.append(((u,v), base_width, edge_color, arrow_s, alpha_val))

#     # --- 7. Layout ---
#     if not G.nodes(): pos = {} # Should be caught earlier
#     elif len(G.nodes()) == 1: pos = {list(G.nodes())[0]: np.array([0.5, 0.5])}
#     else: pos = nx.spring_layout(G, k=spring_layout_k, iterations=spring_layout_iterations, seed=seed, dim=2)

#     # --- 8. Drawing Nodes ---
#     plt.figure(figsize=figure_size, facecolor="white")
#     nx.draw_networkx_nodes(
#         G, pos, node_size=node_sizes_list, node_color=node_colors_list,
#         alpha=0.9, edgecolors="gray", linewidths=1.0
#     )

#     # --- 9. Drawing Edges ---
#     for (u,v), width, color, arr_size, alpha_val in edge_styles_data:
#         # Calculate margins based on actual node sizes at positions
#         # This is an approximation for node radius in plot coordinates
#         # These margins help arrows not to be drawn right from/to the center of large nodes
#         # This part is experimental and might need tuning based on plot scale and node sizes
#         # For simplicity, using fixed small value or scaled from base size for margin
#         source_node_size_attr = node_attrs.get(u, {"size": node_base_size})["size"]
#         target_node_size_attr = node_attrs.get(v, {"size": node_base_size})["size"]
#         # Heuristic for margin, may need adjustment if node sizes vary extremely
#         margin_scale_factor = 30000 # Adjust this based on typical node_base_size and figure size
        
#         source_margin = np.sqrt(source_node_size_attr) / np.sqrt(margin_scale_factor)
#         target_margin = np.sqrt(target_node_size_attr) / np.sqrt(margin_scale_factor)


#         nx.draw_networkx_edges(
#             G, pos, edgelist=[(u,v)], width=width, edge_color=color,
#             arrowsize=arr_size, arrowstyle="-|>", connectionstyle="arc3,rad=0.1",
#             alpha=alpha_val,
#             # The min_source_margin and min_target_margin are relative to the data coordinates.
#             # Their effect depends on the scale of `pos` and node sizes.
#             # Using a more direct scaling:
#             # min_source_margin = 0.01 + 0.00001 * source_node_size_attr, # Experiment
#             # min_target_margin = 0.01 + 0.00001 * target_node_size_attr  # Experiment
#         )
    
#     # --- 10. Drawing Node Labels ---
#     if draw_node_labels:
#         label_pos_adjusted = {
#             node: (coords[0], coords[1] + node_label_y_offset_factor * (np.sqrt(node_attrs[node]["size"] / (np.pi * node_base_size))))
#             for node, coords in pos.items() if node in node_attrs # Ensure node exists
#         }
#         nx.draw_networkx_labels(
#             G, label_pos_adjusted, labels=node_labels_dict, font_size=node_font_size,
#             font_family="sans-serif", font_weight="normal", font_color="#333333"
#         )

#     # --- 11. Drawing Edge Labels ---
#     if draw_edge_labels and G.edges():
#         final_edge_labels = {}
#         for u, v in G.edges():
#             effect = edge_effects.get((u, v), 0.0)
#             is_u_target = node_attrs.get(u,{}).get("is_target",False)
#             is_v_target = node_attrs.get(v,{}).get("is_target",False)
#             is_target_edge = is_u_target or is_v_target

#             if abs(effect) >= edge_label_display_threshold or \
#                (is_target_edge and abs(effect) > 0.001):
#                 final_edge_labels[(u, v)] = f"{effect:.2f}"

#         nx.draw_networkx_edge_labels(
#             G, pos, edge_labels=final_edge_labels,
#             font_size=max(7, int(node_font_size * edge_font_size_ratio)),
#             font_color="#222222",
#             bbox=dict(facecolor="white", alpha=0.75, boxstyle="round,pad=0.2", lw=0.3, edgecolor='lightgrey'),
#             label_pos=0.5, rotate=False
#         )

#     # --- 12. Plot Aesthetics ---
#     plt.title("Causal Relationships Graph", fontsize=16, fontweight="bold", pad=20, color="#111111")
#     plt.axis("off")
    
#     # --- 13. Simplified Legend ---
#     legend_elements = [
#         plt.Line2D([0], [0], color=POSITIVE_EFFECT_COLOR, lw=3, label="Positive Effect"),
#         plt.Line2D([0], [0], color=NEGATIVE_EFFECT_COLOR, lw=3, label="Negative Effect")
#     ]
#     # Check if any neutral edges were actually drawn to decide if to add to legend
#     if any(style_data[2] == NEUTRAL_EFFECT_COLOR for _, _, style_data, _, _ in edge_styles_data if style_data):
#          legend_elements.append(plt.Line2D([0], [0], color=NEUTRAL_EFFECT_COLOR, lw=3, label="Neutral/Zero Effect"))

#     plt.legend(
#         handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02),
#         ncol=len(legend_elements), fontsize=max(9, int(node_font_size)), frameon=False
#     )
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Apply after legend to consider its space
#     plt.show()
