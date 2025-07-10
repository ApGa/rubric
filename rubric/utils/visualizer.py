"""Visualization utilities for rubric trees."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.node import RubricNode
from ..core.scorer import FunctionScorer, LLMScorer, ScriptScorer
from ..core.tree import RubricTree

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class RubricTreeVisualizer:
    """Creates interactive visualizations for rubric trees."""

    def __init__(self) -> None:
        """Initialize the visualizer."""
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Install with: pip install plotly")
        if not NETWORKX_AVAILABLE:
            print("Warning: NetworkX not available. Install with: pip install networkx")

    def visualize_tree_plotly(
        self,
        tree: RubricTree,
        show_scores: bool = False,
        layout: str = "hierarchical",
        width: int = 1600,
        height: int = 1000,
        title: Optional[str] = None,
    ) -> Optional["go.Figure"]:
        """Create an interactive tree visualization using Plotly.

        Args:
            tree: The rubric tree to visualize.
            show_scores: Whether to display scores on the nodes.
            layout: The layout algorithm to use ('hierarchical', 'circular', 'spring').
            width: The width of the plot in pixels.
            height: The height of the plot in pixels.
            title: The title of the plot.

        Returns:
            Plotly figure object, or None if Plotly is not available.
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Please install with: pip install plotly")
            return None

        # Calculate positions based on layout
        if layout == "hierarchical":
            positions = self._calculate_tree_positions(tree)
        elif layout == "circular":
            positions = self._calculate_circular_positions(tree)
        elif layout == "spring":
            if not NETWORKX_AVAILABLE:
                print(
                    "NetworkX not available for spring layout. "
                    "Please install with: pip install networkx"
                )
                return None
            positions = self._calculate_spring_positions(tree)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        # Prepare data for plotting
        edge_x, edge_y = self._create_edges(tree, positions)
        node_data = self._prepare_node_data(tree, positions, show_scores)

        # Create the plot figure
        fig = go.Figure()

        # Add edges with better styling
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="#B0C4DE", width=2),
                hoverinfo="none",
                showlegend=False,
            )
        )

        # Add nodes with enhanced styling for better readability
        fig.add_trace(
            go.Scatter(
                x=node_data["x"],
                y=node_data["y"],
                mode="markers+text",
                marker=dict(
                    size=node_data["sizes"],
                    color=node_data["colors"],
                    line=dict(width=3, color=node_data["border_colors"]),
                    symbol=node_data["symbols"],
                ),
                text=node_data["labels"],
                textposition="top center",
                textfont=dict(size=12, color="#333333"),
                hovertext=node_data["hover_text"],
                hoverinfo="text",
                showlegend=False,
            )
        )

        # Update layout with modern styling
        plot_title = title or f"Rubric Tree: {tree.root.name}"
        fig.update_layout(
            title=dict(
                text=plot_title,
                font=dict(size=18, family="Arial, sans-serif", color="#2C3E50"),
                x=0.5,
                xanchor="center",
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=60, l=20, r=20, t=60),
            annotations=[
                dict(
                    text='<span style="color:#C71585;">●</span> Critical   '
                    '<span style="color:#87CEEB;">●</span> Non-Critical<br>'
                    "<span style=\"font-family: 'Arial', sans-serif; "
                    'font-size: 1.5em; color: #333333">◆</span> Function Scorer   '
                    '<span style="color: #333333">●</span> LLM Scorer',
                    align="left",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.98,
                    y=0.98,
                    xanchor="right",
                    yanchor="top",
                    font=dict(size=14, family="Arial, sans-serif", color="#333333"),
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(255, 255, 255, 0.7)",
                )
            ],
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_data["x"]) - 1, max(node_data["x"]) + 1],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_data["y"]) - 1, max(node_data["y"]) + 1],
            ),
            plot_bgcolor="rgba(240, 248, 255, 0.9)",
            paper_bgcolor="#FFFFFF",
            width=width,
            height=height,
        )

        return fig

    def visualize_as_json(self, tree: RubricTree, file_path: Path) -> None:
        """Save the rubric tree as a JSON file."""
        tree_dict = tree.to_dict()
        with open(file_path, "w") as f:
            json.dump(tree_dict, f, indent=4)
        print(f"Rubric tree saved to {file_path}")

    def generate_interactive_html(
        self,
        tree: RubricTree,
        file_path: Optional[Path] = None,
        show_scores: bool = False,
    ) -> str:
        """Generate an interactive D3.js visualization in an HTML file."""
        if file_path is None:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
                file_path = Path(f.name)

        # Create D3 data and get JavaScript code
        tree_data = self._create_d3_data(tree, show_scores)
        js_code = self._get_javascript_code(tree_data, show_scores)

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rubric Tree</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                {self._get_css_code()}
            </style>
        </head>
        <body>
            <svg width="1200" height="800"></svg>
            <script>
                {js_code}
            </script>
        </body>
        </html>
        """

        # Write to file
        with open(file_path, "w") as f:
            f.write(html_content)
        print(f"Interactive D3 visualization saved to: {file_path}")
        return str(file_path)

    def _calculate_tree_positions(self, tree: RubricTree) -> Dict[str, Tuple[float, float]]:
        """Calculate positions for nodes in a hierarchical tree layout with better spacing."""
        positions: Dict[str, Tuple[float, float]] = {}
        levels = {}

        # Assign levels (depth) to each node
        def assign_levels(node: RubricNode, level: int = 0) -> None:
            if node.name not in levels:
                levels[node.name] = level
                for child in node.children:
                    assign_levels(child, level + 1)

        assign_levels(tree.root)

        # Group nodes by level
        level_groups: Dict[int, List[str]] = {}
        for name, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(name)

        # Calculate positions
        max_level = max(levels.values()) if levels else 0
        level_height = 10.0

        for level, nodes in level_groups.items():
            y = (max_level - level) * level_height
            num_nodes = len(nodes)

            if num_nodes == 1:
                x_positions = [0.0]
            else:
                # Dynamic horizontal spacing
                base_width = float(max(80, num_nodes * 40))  # Further adjusted horizontal spacing
                if level > 0:
                    base_width *= 1 + level * 0.8

                x_positions = [
                    -base_width / 2 + i * (base_width / (num_nodes - 1)) for i in range(num_nodes)
                ]

            for node_name, x_pos in zip(nodes, x_positions):
                positions[node_name] = (x_pos, y)

        return positions

    def _calculate_circular_positions(self, tree: RubricTree) -> Dict[str, Tuple[float, float]]:
        """Calculate positions for nodes in a circular layout."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for circular layout.")

        G = self._create_networkx_graph(tree)
        pos = nx.circular_layout(G)
        return {name: (float(coords[0]), float(coords[1])) for name, coords in pos.items()}

    def _calculate_spring_positions(self, tree: RubricTree) -> Dict[str, Tuple[float, float]]:
        """Calculate positions using a force-directed spring layout."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for spring layout.")

        G = self._create_networkx_graph(tree)
        pos = nx.spring_layout(G, k=0.9, iterations=100, seed=42)
        return {name: (float(coords[0]), float(coords[1])) for name, coords in pos.items()}

    def _create_networkx_graph(self, tree: RubricTree) -> "nx.Graph":
        """Create a NetworkX graph from the RubricTree."""
        G = nx.Graph()
        all_nodes = tree.get_all_nodes()
        for node in all_nodes:
            G.add_node(node.name)
            for child in node.children:
                G.add_edge(node.name, child.name)
        return G

    def _create_edges(
        self, tree: RubricTree, positions: Dict[str, Tuple[float, float]]
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """Create edges (lines) connecting nodes."""
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []

        def add_edges(node: RubricNode) -> None:
            if node.name not in positions:
                return
            x0, y0 = positions[node.name]
            for child in node.children:
                if child.name in positions:
                    x1, y1 = positions[child.name]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                add_edges(child)

        add_edges(tree.root)
        return edge_x, edge_y

    def _prepare_node_data(
        self, tree: RubricTree, positions: Dict[str, Tuple[float, float]], show_scores: bool
    ) -> Dict[str, List]:
        """Prepare data for Plotly nodes, including labels, sizes, and colors."""
        node_data: Dict[str, List] = {
            "x": [],
            "y": [],
            "labels": [],
            "sizes": [],
            "colors": [],
            "border_colors": [],
            "symbols": [],
            "hover_text": [],
        }

        all_nodes = tree.get_all_nodes()

        for node in all_nodes:
            if node.name not in positions:
                continue

            x, y = positions[node.name]
            node_data["x"].append(x)
            node_data["y"].append(y)

            # Add labels, wrapping long descriptions
            node_name = f"Overall: {node.name}" if node is tree.root else node.name
            if node.description:
                wrapped_description = self._wrap_text(node.description, 40)
                label = f"<b>{node_name}</b><br>{wrapped_description}"
            else:
                label = f"<b>{node_name}</b>"

            if show_scores and node.score is not None:
                label += f"<br><i>Score: {node.score:.2f}</i>"

            node_data["labels"].append(label)

            # Node size based on content length
            size = 25 + len(label) * 0.1
            node_data["sizes"].append(size)

            # Node color based on criticality
            color = "#C71585" if node.is_critical else "#87CEEB"
            node_data["colors"].append(color)

            # Border color can be kept for additional info, but not in legend
            if node.scorer and isinstance(node.scorer, FunctionScorer):
                border_color = "#F1C40F"  # Yellow for Function
            elif node.scorer and isinstance(node.scorer, LLMScorer):
                border_color = "#E67E22"  # Orange for LLM
            elif node.scorer and isinstance(node.scorer, ScriptScorer):
                border_color = "#1ABC9C"  # Green for script
            else:
                border_color = "#2C3E50"  # Dark blue for others

            node_data["border_colors"].append(border_color)

            # Node symbol based on scorer type
            symbol = "square"  # Default for parent nodes
            if node.scorer:
                if isinstance(node.scorer, FunctionScorer):
                    symbol = "diamond"
                elif isinstance(node.scorer, LLMScorer):
                    symbol = "circle"
                elif isinstance(node.scorer, ScriptScorer):
                    symbol = "square"  # Or another shape if you want to distinguish

            node_data["symbols"].append(symbol)

            # Create hover text with full details
            hover_text = (
                f"<b>{node_name}</b><br>"
                f"Description: {node.description or 'N/A'}<br>"
                f"Critical: {'Yes' if node.is_critical else 'No'}"
            )

            if node.scorer:
                scorer_type = type(node.scorer).__name__
                hover_text += f"<br>Scorer: {scorer_type}"
                if isinstance(node.scorer, LLMScorer):
                    prompt = self._wrap_text(node.scorer.prompt_template, 80)
                    hover_text += f"<br><br><b>Prompt:</b><br>{prompt}"
                elif isinstance(node.scorer, ScriptScorer):
                    script = self._wrap_text(node.scorer.script_content, 80)
                    hover_text += f"<br><br><b>Script:</b><br><pre>{script}</pre>"
                elif isinstance(node.scorer, FunctionScorer):
                    code = self._wrap_text(node.scorer.function_code, 80)
                    hover_text += f"<br><br><b>Function:</b><br><pre>{code}</pre>"

            if show_scores and node.score is not None:
                hover_text += f"<br><b>Score: {node.score:.2f}</b>"

            node_data["hover_text"].append(hover_text)

        return node_data

    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to a specified width, respecting HTML tags."""
        lines = text.split("\n")
        wrapped_lines = []
        for line in lines:
            words = line.split(" ")
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 > width:
                    wrapped_lines.append(current_line)
                    current_line = word
                else:
                    current_line += " " + word
            wrapped_lines.append(current_line.strip())
        return "<br>".join(wrapped_lines)

    def _get_css_code(self) -> str:
        """Return the CSS code for the D3 visualization."""
        return """
        .node circle {
            stroke-width: 3px;
        }
        .node.critical circle {
            stroke: #C71585;
            fill: #FFB6C1;
        }
        .node.non-critical circle {
            stroke: #4682B4;
            fill: #B0E0E6;
        }
        .link {
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }
        .node text {
            font: 12px sans-serif;
        }
        .tooltip {
            position: absolute;
            text-align: center;
            width: auto;
            height: auto;
            padding: 8px;
            font: 12px sans-serif;
            background: lightsteelblue;
            border: 0px;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .tooltip .critical {
            color: #C71585;
            font-weight: bold;
        }
        .tooltip .score {
            font-weight: bold;
            font-size: 1.1em;
        }
        .tooltip .scorer {
            font-style: italic;
        }
        """

    def _get_javascript_code(self, tree_data: str, show_scores: bool) -> str:
        """Return the JavaScript code for rendering the D3 visualization."""
        js_code = f"""
         document.addEventListener("DOMContentLoaded", function() {{
             const data = {tree_data};
             const svg = d3.select("svg");
             const width = +svg.attr("width");
             const height = +svg.attr("height");
             const g = svg.append("g").attr("transform", "translate(40,0)");

             const tree = d3.tree().size([height - 40, width - 160]);
             const root = d3.hierarchy(data);
             tree(root);

             const link = g.selectAll(".link")
                 .data(root.links())
                 .enter().append("path")
                 .attr("class", "link")
                 .attr("d", d3.linkHorizontal()
                     .x(d => d.y)
                     .y(d => d.x));

             const node = g.selectAll(".node")
                 .data(root.descendants())
                 .enter().append("g")
                 .attr("class", d => `node ${{d.data.is_critical ? 'critical' : 'non-critical'}}`)
                 .attr("transform", d => `translate(${{d.y}},${{d.x}})`);

             node.append("circle")
                 .attr("r", 10);

             node.append("text")
                 .attr("dy", "0.31em")
                 .attr("x", d => d.children ? -12 : 12)
                 .style("text-anchor", d => d.children ? "end" : "start")
                 .text(d => d.data.name);

             const tooltip = d3.select("body").append("div")
                 .attr("class", "tooltip");

             node.on("mouseover", (event, d) => {{
                 tooltip.transition().duration(200).style("opacity", .9);
                 let content = `<h3>${{d.data.name}}</h3>`;
                 if (d.data.description) {{
                     content += `<p>${{d.data.description}}</p>`;
                 }}
                 if (d.data.is_critical) {{
                     content += `<p class="critical">Critical Criterion</p>`;
                 }}
                 if ({str(show_scores).lower()} && d.data.score !== undefined) {{
                     content += `<p>Score: <span class="score">${{d.data.score}}</span></p>`;
                 }}
                 if (d.data.scorer_type) {{
                     content += `<p class="scorer">Scorer: ${{d.data.scorer_type}}</p>`;
                 }}
                 tooltip.html(content);
             }})
             .on("mousemove", (event) => {{
                 tooltip.style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 28) + "px");
             }})
             .on("mouseout", () => {{
                 tooltip.style("opacity", 0);
             }});

             // Zoom and pan functionality
             const zoom = d3.zoom().on("zoom", (event) => {{
                 g.attr("transform", event.transform);
             }});
             svg.call(zoom);

         }});
         """
        return js_code

    def _create_d3_data(self, tree: RubricTree, show_scores: bool) -> str:
        """Create D3 data for the visualization."""
        # This method is a placeholder and should be implemented to return the
        # correct D3 data format
        # based on the tree structure and the show_scores flag.
        # For now, it returns an empty string as a placeholder.
        return ""
