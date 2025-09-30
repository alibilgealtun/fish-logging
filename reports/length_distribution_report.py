"""
Length Distribution Report Generator

Generates exportable reports (PDF) with graphs and raw data for
length distribution by hauls using logs/hauls/logs.xlsx.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
# Make matplotlib optional at import time to avoid hard failures in environments without it
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore
import seaborn as sns

# Use Plotly for interactive, modern charts
try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - optional at import time; UI will guard usage
    px = None  # type: ignore
    go = None  # type: ignore

# Limit UI chart choices to the two Plotly-based charts requested
CHART_TYPES = {
    "species_pie": "Species Composition (Pie)",
    "species_avg_length_bar": "Average Length by Species (Bar)",
    "species_count_bar": "Total Count by Species (Bar)",
    "species_length_distribution": "Length Distribution (Select Species)",
}


@dataclass
class HaulSummary:
    haul_id: str
    date: str
    boat: str
    species_count: int
    total_fish: int
    avg_length: float
    std_length: float
    min_length: float
    max_length: float
    confidence_avg: float


class ReportExporter:
    def export(self, report_data: Dict, output_path: Path) -> None:  # pragma: no cover
        raise NotImplementedError


class PDFExporter(ReportExporter):
    def export(self, report_data: Dict, output_path: Path) -> None:
        # Import inside method and guard availability
        try:
            from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
            import matplotlib.pyplot as _plt  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Matplotlib is required for PDF export. Install with: pip install matplotlib") from e
        from io import BytesIO

        figures = report_data.get("figures", [])
        with PdfPages(output_path) as pdf:  # type: ignore
            for fig in figures:
                try:
                    # If it's a Plotly figure, rasterize to PNG via Kaleido
                    if go is not None and isinstance(fig, go.Figure):  # type: ignore[arg-type]
                        png_bytes = fig.to_image(format="png", scale=2)
                        # Place image onto a matplotlib figure page
                        img_buf = BytesIO(png_bytes)
                        img = _plt.imread(img_buf, format='png')
                        h, w = img.shape[:2]
                        # Create figure sized to image aspect
                        dpi = 100
                        fig_w = w / dpi
                        fig_h = h / dpi
                        mfig = _plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                        ax = mfig.add_axes([0, 0, 1, 1])
                        ax.axis('off')
                        ax.imshow(img)
                        pdf.savefig(mfig, bbox_inches="tight", pad_inches=0)
                        _plt.close(mfig)
                    else:
                        # Fallback: assume matplotlib Figure
                        pdf.savefig(fig, bbox_inches="tight")
                except Exception:
                    # As a last resort, try to convert via generic PNG path
                    try:
                        png_bytes = getattr(fig, "to_image")(format="png", scale=2)  # type: ignore[misc]
                        img_buf = BytesIO(png_bytes)
                        img = _plt.imread(img_buf, format='png')
                        mfig = _plt.figure(figsize=(8, 6))
                        ax = mfig.add_axes([0, 0, 1, 1])
                        ax.axis('off')
                        ax.imshow(img)
                        pdf.savefig(mfig, bbox_inches="tight", pad_inches=0)
                        _plt.close(mfig)
                    except Exception:
                        # Skip figures that cannot be exported
                        continue
            info = pdf.infodict()
            info["Title"] = "Fish Length Distribution Report"
            info["Author"] = "Fish Logging System"
            info["Subject"] = "Length Distribution Analysis by Hauls"
            info["Keywords"] = "Fish, Length, Distribution, Hauls, Analysis"
            info["CreationDate"] = datetime.now()


class LengthDistributionReportGenerator:
    def __init__(self, data_path: Optional[Path] = None) -> None:
        self.data_path = Path(data_path) if data_path else Path("logs/hauls/logs.xlsx")
        self.exporters: Dict[str, ReportExporter] = {
            "pdf": PDFExporter(),
        }
        # Modern plot styling for Matplotlib exports
        self._modern_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                               '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f']
        self._setup_modern_style()
        # Plotly color mapping
        self._plotly_color_discrete = self._modern_colors

    def _setup_modern_style(self) -> None:
        """Configure modern matplotlib + seaborn styling."""
        if plt is None:  # If matplotlib isn't available, skip style setup gracefully
            return
        try:
            plt.style.use("default")  # Start with clean slate

            # Set matplotlib parameters for modern look
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': '#fafafa',
                'axes.edgecolor': '#e0e0e0',
                'axes.linewidth': 1.2,
                'axes.grid': True,
                'axes.axisbelow': True,
                'axes.labelcolor': '#2c3e50',
                'axes.titlesize': 16,
                'axes.titleweight': 'bold',
                'axes.titlepad': 16,
                'axes.labelsize': 12,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': True,
                'axes.spines.bottom': True,
                'grid.color': '#e8e8e8',
                'grid.linewidth': 0.8,
                'grid.alpha': 0.9,
                'xtick.color': '#7f8c8d',
                'ytick.color': '#7f8c8d',
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.frameon': True,
                'legend.fancybox': True,
                'legend.shadow': False,
                'legend.framealpha': 1.0,
                'legend.facecolor': 'white',
                'legend.edgecolor': '#e0e0e0',
                'font.family': ['SF Pro Display', 'Segoe UI', 'system-ui', 'sans-serif'],
                'font.size': 11,
                'figure.autolayout': True,
            })

            # Set the modern color cycle
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self._modern_colors)
        except Exception:
            pass
        # Seaborn theme
        try:
            sns.set_theme(style="whitegrid", context="talk", palette=self._modern_colors)
        except Exception:
            # Fallback if seaborn version differs
            try:
                sns.set_style("whitegrid")
            except Exception:
                pass

    # Data loading/cleaning
    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_excel(self.data_path)
        required = ["Date", "Time", "Boat", "Species", "Length (cm)", "Confidence"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # Normalize types
        # Parse dates to a consistent dtype and drop invalid rows
        date_parsed = pd.to_datetime(df["Date"], errors="coerce")
        df = df[~date_parsed.isna()].copy()
        df.loc[:, "Date"] = date_parsed.loc[df.index].dt.date  # store as date objects for stable min/max
        # Numeric columns
        df["Length (cm)"] = pd.to_numeric(df["Length (cm)"], errors="coerce")
        df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
        df = df.dropna(subset=["Length (cm)", "Confidence"])  # remove invalid rows
        df = df[df["Length (cm)"] > 0]
        # Ensure Boat and Species are strings
        df["Boat"] = df["Boat"].astype(str).fillna("")
        df["Species"] = df["Species"].astype(str).fillna("")
        # Haul identifier: use ISO date string + boat
        df["Haul_ID"] = df["Date"].astype(str) + "_" + df["Boat"]
        return df

    # Analytics
    def generate_haul_summaries(self, df: pd.DataFrame) -> List[HaulSummary]:
        summaries: List[HaulSummary] = []
        for hid in df["Haul_ID"].unique():
            part = df[df["Haul_ID"] == hid]
            summaries.append(
                HaulSummary(
                    haul_id=hid,
                    date=str(part["Date"].iloc[0]),
                    boat=str(part["Boat"].iloc[0]),
                    species_count=int(part["Species"].nunique()),
                    total_fish=int(len(part)),
                    avg_length=float(part["Length (cm)"].mean()),
                    std_length=float(part["Length (cm)"].std() or 0.0),
                    min_length=float(part["Length (cm)"].min()),
                    max_length=float(part["Length (cm)"].max()),
                    confidence_avg=float(part["Confidence"].mean()),
                )
            )
        return summaries

    def species_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby(["Haul_ID", "Species"]).agg(
            **{
                "Count": ("Length (cm)", "count"),
                "Mean_Length": ("Length (cm)", "mean"),
                "Std_Length": ("Length (cm)", "std"),
                "Min_Length": ("Length (cm)", "min"),
                "Max_Length": ("Length (cm)", "max"),
                "Avg_Confidence": ("Confidence", "mean"),
            }
        )
        return agg.reset_index().round(2)

    # Plotly charts for UI visualization
    def create_plotly_chart(self, df: pd.DataFrame, chart_key: str):
        if px is None or go is None:
            raise RuntimeError("Plotly is not installed. Please install 'plotly' and 'kaleido'.")
        key = chart_key or "species_pie"
        if key == "species_pie":
            agg = (
                df.groupby("Species").size().reset_index(name="Count").sort_values("Count", ascending=False)
            )
            if agg.empty:
                return go.Figure()
            total = agg["Count"].sum()
            agg["Percent"] = (agg["Count"] / total * 100).round(1)
            fig = px.pie(
                agg,
                names="Species",
                values="Count",
                color="Species",
                color_discrete_sequence=self._plotly_color_discrete,
                hole=0.5,
            )
            fig.update_traces(textposition='inside', texttemplate="%{customdata:.1f}%", hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percent: %{customdata:.1f}%<extra></extra>", customdata=agg["Percent"])  # type: ignore[index]
            fig.update_layout(
                title_text="Species Composition",
                legend_title_text="Species",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            return fig

        if key == "species_avg_length_bar":
            agg = (
                df.groupby("Species")["Length (cm)"].agg(Avg_Length="mean", Count="count").reset_index()
            )
            if agg.empty:
                return go.Figure()
            agg["Avg_Length"] = agg["Avg_Length"].round(2)
            agg = agg.sort_values("Avg_Length", ascending=True)
            fig = px.bar(
                agg,
                x="Avg_Length",
                y="Species",
                orientation="h",
                color="Species",
                color_discrete_sequence=self._plotly_color_discrete,
                text="Avg_Length",
            )
            fig.update_traces(texttemplate="%{x:.2f} cm", textposition="outside", cliponaxis=False)
            fig.update_layout(
                title_text="Average Length by Species",
                xaxis_title="Average Length (cm)",
                yaxis_title="Species",
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
                height=max(320, 22 * len(agg)),
            )
            return fig

        if key == "species_count_bar":
            agg = df.groupby("Species").size().reset_index(name="Count").sort_values("Count", ascending=True)
            if agg.empty:
                return go.Figure()
            fig = px.bar(
                agg,
                x="Count",
                y="Species",
                orientation="h",
                color="Species",
                color_discrete_sequence=self._plotly_color_discrete,
                text="Count",
            )
            fig.update_traces(texttemplate="%{x}", textposition="outside", cliponaxis=False)
            fig.update_layout(
                title_text="Total Count by Species",
                xaxis_title="Count",
                yaxis_title="Species",
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
                height=max(320, 22 * len(agg)),
            )
            return fig
        # Fallback
        return self.create_plotly_chart(df, "species_pie")

    def create_species_length_chart(self, df: pd.DataFrame, species: str):
        """Create an interactive Plotly histogram for a single species' length distribution.

        Returns an empty figure if there is no data for the species.
        """
        if px is None or go is None:
            raise RuntimeError("Plotly is not installed. Please install 'plotly' and 'kaleido'.")
        try:
            sp = str(species).strip()
        except Exception:
            sp = species
        if not sp:
            return go.Figure()
        part = df[df["Species"] == sp]
        if part.empty:
            # Empty figure with a friendly annotation
            fig = go.Figure()
            fig.update_layout(
                title_text=f"No data for species: {sp}",
                margin=dict(l=10, r=10, t=50, b=10),
                height=380,
            )
            return fig
        # Basic histogram with box marginal and mean line
        fig = px.histogram(
            part,
            x="Length (cm)",
            nbins=20,
            opacity=0.85,
            marginal="box",
            color_discrete_sequence=[self._plotly_color_discrete[0]],
        )
        mean_val = float(part["Length (cm)"].mean())
        min_v = float(part["Length (cm)"].min())
        max_v = float(part["Length (cm)"].max())
        count = int(len(part))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#e74c3c", annotation_text=f"Mean: {mean_val:.2f} cm", annotation_position="top")
        fig.update_layout(
            title_text=f"Length Distribution - {sp} (n={count}, min={min_v:.1f}, mean={mean_val:.1f}, max={max_v:.1f})",
            xaxis_title="Length (cm)",
            yaxis_title="Frequency",
            margin=dict(l=10, r=10, t=60, b=10),
            height=420,
        )
        return fig

    # Plots assembled for export (now using Plotly)
    def create_figures(self, df: pd.DataFrame) -> List[object]:
        if px is None or go is None:
            return []
        figs: List[object] = []
        # Only include charts that exist in ReportWidget (species_pie, avg_length_bar, count_bar)
        for key in ("species_pie", "species_avg_length_bar", "species_count_bar"):
            try:
                fig = self.create_plotly_chart(df, key)
                figs.append(fig)
            except Exception:
                # Skip any chart that fails to render
                continue
        return figs

    # Orchestration
    def generate_report(self, output_dir: Path, formats: Optional[List[str]] = None) -> Dict:
        fmts = formats or ["pdf"]
        df = self.load_data()
        if df.empty:
            raise ValueError("No valid data found in the dataset")
        summaries = self.generate_haul_summaries(df)
        species_df = self.species_distribution(df)
        figures = self.create_figures(df)
        report_data: Dict = {
            "raw_data": df,
            "haul_summaries": summaries,
            "species_distribution": species_df,
            "figures": figures,
            "metadata": {
                "generation_date": datetime.now(),
                "total_hauls": int(df["Haul_ID"].nunique()),
                "total_fish": int(len(df)),
                "species_count": int(df["Species"].nunique()),
                "date_range": (str(df["Date"].min()), str(df["Date"].max())),
            },
        }
        # Export files
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported: Dict[str, Path] = {}
        for ft in fmts:
            exp = self.exporters.get(ft)
            if not exp:
                continue
            ext = ft
            out_path = output_dir / f"length_distribution_report_{stamp}.{ext}"
            exp.export(report_data, out_path)
            exported[ft] = out_path
        # Close any matplotlib figures we created internally
        try:
            if plt is not None:
                plt.close('all')
        except Exception:
            pass

        report_data["exported_files"] = exported
        return report_data
