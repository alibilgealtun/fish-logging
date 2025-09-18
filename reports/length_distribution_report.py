"""
Length Distribution Report Generator

Generates exportable reports (PDF/Excel) with graphs and raw data for
length distribution by hauls using logs/hauls/logs.xlsx.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
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
        from matplotlib.backends.backend_pdf import PdfPages
        from io import BytesIO

        figures = report_data.get("figures", [])
        with PdfPages(output_path) as pdf:
            for fig in figures:
                try:
                    # If it's a Plotly figure, rasterize to PNG via Kaleido
                    if go is not None and isinstance(fig, go.Figure):  # type: ignore[arg-type]
                        png_bytes = fig.to_image(format="png", scale=2)
                        # Place image onto a matplotlib figure page
                        img_buf = BytesIO(png_bytes)
                        img = plt.imread(img_buf, format='png')
                        h, w = img.shape[:2]
                        # Create figure sized to image aspect
                        dpi = 100
                        fig_w = w / dpi
                        fig_h = h / dpi
                        mfig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                        ax = mfig.add_axes([0, 0, 1, 1])
                        ax.axis('off')
                        ax.imshow(img)
                        pdf.savefig(mfig, bbox_inches="tight", pad_inches=0)
                        plt.close(mfig)
                    else:
                        # Fallback: assume matplotlib Figure
                        pdf.savefig(fig, bbox_inches="tight")
                except Exception:
                    # As a last resort, try to convert via generic PNG path
                    try:
                        png_bytes = getattr(fig, "to_image")(format="png", scale=2)  # type: ignore[misc]
                        img_buf = BytesIO(png_bytes)
                        img = plt.imread(img_buf, format='png')
                        mfig = plt.figure(figsize=(8, 6))
                        ax = mfig.add_axes([0, 0, 1, 1])
                        ax.axis('off')
                        ax.imshow(img)
                        pdf.savefig(mfig, bbox_inches="tight", pad_inches=0)
                        plt.close(mfig)
                    except Exception:
                        # Skip figures that cannot be exported
                        continue
            info = pdf.infodict()
            info["Title"] = "Fish Length Distribution Report"
            info["Author"] = "Fish Logging System"
            info["Subject"] = "Length Distribution Analysis by Hauls"
            info["Keywords"] = "Fish, Length, Distribution, Hauls, Analysis"
            info["CreationDate"] = datetime.now()


class ExcelExporter(ReportExporter):
    def export(self, report_data: Dict, output_path: Path) -> None:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            if "raw_data" in report_data:
                report_data["raw_data"].to_excel(writer, sheet_name="Raw Data", index=False)
            if "haul_summaries" in report_data:
                summary_df = pd.DataFrame([
                    {
                        "Haul ID": h.haul_id,
                        "Date": h.date,
                        "Boat": h.boat,
                        "Species Count": h.species_count,
                        "Total Fish": h.total_fish,
                        "Average Length (cm)": h.avg_length,
                        "Std Length (cm)": h.std_length,
                        "Min Length (cm)": h.min_length,
                        "Max Length (cm)": h.max_length,
                        "Avg Confidence": h.confidence_avg,
                    }
                    for h in report_data["haul_summaries"]
                ])
                summary_df.to_excel(writer, sheet_name="Haul Summary", index=False)
            if "species_distribution" in report_data:
                report_data["species_distribution"].to_excel(
                    writer, sheet_name="Species Distribution", index=False
                )


class LengthDistributionReportGenerator:
    def __init__(self, data_path: Optional[Path] = None) -> None:
        self.data_path = Path(data_path) if data_path else Path("logs/hauls/logs.xlsx")
        self.exporters: Dict[str, ReportExporter] = {
            "pdf": PDFExporter(),
            "excel": ExcelExporter(),
        }
        # Modern plot styling for Matplotlib exports
        self._modern_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                               '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f']
        self._setup_modern_style()
        # Plotly color mapping
        self._plotly_color_discrete = self._modern_colors

    def _setup_modern_style(self) -> None:
        """Configure modern matplotlib + seaborn styling."""
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

        # Seaborn theme
        try:
            sns.set_theme(style="whitegrid", context="talk", palette=self._modern_colors)
        except Exception:
            # Fallback if seaborn version differs
            sns.set_style("whitegrid")

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
        df["Length (cm)"] = pd.to_numeric(df["Length (cm)"], errors="coerce")
        df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
        df = df.dropna(subset=["Length (cm)", "Confidence"])  # remove invalid rows
        df = df[df["Length (cm)"] > 0]
        # Haul identifier
        df["Haul_ID"] = df["Date"].astype(str) + "_" + df["Boat"].astype(str)
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
        # Fallback
        return self.create_plotly_chart(df, "species_pie")

    # Plots assembled for export (now using Plotly)
    def create_figures(self, df: pd.DataFrame) -> List[object]:
        if px is None or go is None:
            # Fallback to empty list if Plotly isn't available
            return []
        figs: List[object] = []

        # 1. Overall histogram
        fig1 = px.histogram(
            df,
            x="Length (cm)",
            nbins=30,
            marginal="rug",
            opacity=0.85,
            color_discrete_sequence=[self._plotly_color_discrete[0]],
        )
        fig1.update_layout(title_text="Overall Fish Length Distribution", xaxis_title="Length (cm)", yaxis_title="Frequency")
        figs.append(fig1)

        # 2. Boxplot by haul
        if df["Haul_ID"].nunique() > 0:
            fig2 = px.box(
                df,
                x="Haul_ID",
                y="Length (cm)",
                color="Haul_ID",
                color_discrete_sequence=self._plotly_color_discrete,
            )
            fig2.update_layout(title_text="Length Distribution by Haul", xaxis_title="Haul ID", yaxis_title="Length (cm)", showlegend=False)
            figs.append(fig2)

        # 3. Hist by species
        if df["Species"].nunique() > 1:
            fig3 = px.histogram(
                df,
                x="Length (cm)",
                color="Species",
                nbins=20,
                barmode="overlay",
                opacity=0.65,
                color_discrete_sequence=self._plotly_color_discrete,
            )
            fig3.update_layout(title_text="Length Distribution by Species", xaxis_title="Length (cm)", yaxis_title="Frequency")
            figs.append(fig3)

        # 4. Violin by haul
        if df["Haul_ID"].nunique() > 1:
            fig4 = px.violin(
                df,
                x="Haul_ID",
                y="Length (cm)",
                color="Haul_ID",
                box=True,
                points=False,
                color_discrete_sequence=self._plotly_color_discrete,
            )
            fig4.update_layout(title_text="Length Distribution Density by Haul", xaxis_title="Haul ID", yaxis_title="Length (cm)", showlegend=False)
            figs.append(fig4)

        # 5. Bar mean by haul with sd error bars
        means = df.groupby("Haul_ID")["Length (cm)"].mean().reset_index(name="Mean")
        stds = df.groupby("Haul_ID")["Length (cm)"].std().reset_index(name="Std")
        merged = means.merge(stds, on="Haul_ID", how="left").fillna({"Std": 0})
        fig5 = go.Figure(
            data=[
                go.Bar(
                    x=merged["Haul_ID"],
                    y=merged["Mean"],
                    error_y=dict(type='data', array=merged["Std"], visible=True, thickness=1.2),
                    marker_color=self._plotly_color_discrete,
                )
            ]
        )
        fig5.update_layout(title_text="Average Fish Length by Haul", xaxis_title="Haul ID", yaxis_title="Average Length (cm)")
        figs.append(fig5)

        # 6. Species composition pie (donut)
        species_counts = df["Species"].value_counts()
        if not species_counts.empty:
            agg = species_counts.reset_index()
            agg.columns = ["Species", "Count"]
            fig6 = px.pie(
                agg,
                names="Species",
                values="Count",
                color="Species",
                color_discrete_sequence=self._plotly_color_discrete,
                hole=0.4,
            )
            fig6.update_traces(textposition='inside', textinfo='percent')
            fig6.update_layout(title_text="Species Composition")
            figs.append(fig6)

        return figs

    # Orchestration
    def generate_report(self, output_dir: Path, formats: Optional[List[str]] = None) -> Dict:
        fmts = formats or ["pdf", "excel"]
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
            ext = 'xlsx' if ft == 'excel' else ft
            out_path = output_dir / f"length_distribution_report_{stamp}.{ext}"
            exp.export(report_data, out_path)
            exported[ft] = out_path
        # Close any matplotlib figures we created internally
        plt.close('all')

        report_data["exported_files"] = exported
        return report_data
