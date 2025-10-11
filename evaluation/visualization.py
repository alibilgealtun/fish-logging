"""Visualization and enhanced reporting for evaluation results.

This module generates academic-quality visualizations and comprehensive
analysis reports from evaluation data.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from loguru import logger

# Optional plotting dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except Exception:
    _PLOTTING_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available; visualizations will be skipped")


class EvaluationVisualizer:
    """Generate visualizations and enhanced reports for evaluation results."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, df: pd.DataFrame, df_agg: pd.DataFrame) -> Dict[str, Any]:
        """Generate all visualizations and enhanced summary.

        Args:
            df: Full results dataframe
            df_agg: Aggregate-only dataframe (for summary stats)

        Returns:
            Dictionary with analysis results and generated file paths
        """
        analysis = {}

        if not _PLOTTING_AVAILABLE:
            logger.warning("Plotting unavailable; skipping visualizations")
            analysis["error"] = "matplotlib/seaborn not installed"
        else:
            try:
                # Generate visualizations
                analysis["heatmap"] = self._generate_accent_model_heatmap(df_agg)
                analysis["error_categorization"] = self._generate_error_categorization(df_agg)
                analysis["speaker_breakdown"] = self._generate_speaker_breakdown(df_agg)
                analysis["accent_correlation"] = self._generate_accent_correlation(df_agg)
                analysis["per_condition"] = self._generate_per_condition_breakdown(df_agg)
                logger.info(f"Generated visualizations in {self.viz_dir}")
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
                analysis["error"] = str(e)

        # Generate comprehensive markdown summary (doesn't require plotting)
        try:
            summary_stats = self._compute_comprehensive_stats(df, df_agg)
            analysis.update(summary_stats)
            self._generate_enhanced_summary(summary_stats)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            analysis["summary_error"] = str(e)

        return analysis

    def _compute_comprehensive_stats(self, df: pd.DataFrame, df_agg: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistics for summary."""
        stats = {}

        # Overall statistics
        stats["total_samples"] = len(df_agg)
        stats["total_passed"] = int(df_agg["numeric_exact_match"].sum())
        stats["total_failed"] = stats["total_samples"] - stats["total_passed"]
        stats["overall_accuracy"] = float(df_agg["numeric_exact_match"].mean()) if len(df_agg) > 0 else 0.0

        # Per-speaker analysis
        speaker_stats = []
        for speaker, grp in df_agg.groupby("speaker_id"):
            total = len(grp)
            passed = int(grp["numeric_exact_match"].sum())
            speaker_stats.append({
                "speaker": speaker,
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy": passed / total if total > 0 else 0.0,
                "accent": grp["accent"].iloc[0] if len(grp) > 0 else "unknown"
            })
        stats["by_speaker"] = sorted(speaker_stats, key=lambda x: x["accuracy"], reverse=True)

        # Per-accent analysis
        accent_stats = []
        for accent, grp in df_agg.groupby("accent"):
            total = len(grp)
            passed = int(grp["numeric_exact_match"].sum())
            accent_stats.append({
                "accent": accent,
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy": passed / total if total > 0 else 0.0
            })
        stats["by_accent"] = sorted(accent_stats, key=lambda x: x["accuracy"], reverse=True)

        # Per-model analysis
        model_stats = []
        for (model, size), grp in df_agg.groupby(["model_name", "model_size"]):
            total = len(grp)
            passed = int(grp["numeric_exact_match"].sum())
            model_stats.append({
                "model": f"{model}/{size}",
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy": passed / total if total > 0 else 0.0,
                "avg_rtf": float(grp["RTF"].mean()) if "RTF" in grp else 0.0,
                "avg_confidence": float(grp["confidence"].mean()) if "confidence" in grp else 0.0
            })
        stats["by_model"] = sorted(model_stats, key=lambda x: x["accuracy"], reverse=True)

        # Error type distribution
        if "error_type" in df_agg.columns:
            error_dist = df_agg["error_type"].value_counts().to_dict()
            stats["error_distribution"] = {str(k): int(v) for k, v in error_dist.items()}
        else:
            stats["error_distribution"] = {}

        return stats

    def _generate_enhanced_summary(self, stats: Dict[str, Any]):
        """Generate enhanced markdown summary with comprehensive analysis."""
        md_lines = [
            "# Comprehensive Evaluation Summary",
            "",
            "## Overall Results",
            "",
            f"**Total Samples:** {stats['total_samples']}  ",
            f"**Passed:** {stats['total_passed']} ({stats['overall_accuracy']*100:.2f}%)  ",
            f"**Failed:** {stats['total_failed']} ({(1-stats['overall_accuracy'])*100:.2f}%)  ",
            "",
            "---",
            "",
            "## 📊 Quick Summary - Key Totals",
            "",
            "### By Speaker",
            ""
        ]

        # Add speaker totals in a concise format
        for s in stats["by_speaker"]:
            status_emoji = "✅" if s['accuracy'] >= 0.95 else "⚠️" if s['accuracy'] >= 0.80 else "❌"
            md_lines.append(
                f"- {status_emoji} **{s['speaker']}** ({s['accent']}): "
                f"{s['passed']}/{s['total']} passed ({s['accuracy']*100:.1f}%)"
            )

        md_lines.extend([
            "",
            "### By Accent",
            ""
        ])

        # Add accent totals
        for a in stats["by_accent"]:
            status_emoji = "✅" if a['accuracy'] >= 0.95 else "⚠️" if a['accuracy'] >= 0.80 else "❌"
            md_lines.append(
                f"- {status_emoji} **{a['accent']}**: "
                f"{a['passed']}/{a['total']} passed ({a['accuracy']*100:.1f}%)"
            )

        md_lines.extend([
            "",
            "### By Model",
            ""
        ])

        # Add model totals
        for m in stats["by_model"]:
            status_emoji = "✅" if m['accuracy'] >= 0.95 else "⚠️" if m['accuracy'] >= 0.80 else "❌"
            md_lines.append(
                f"- {status_emoji} **{m['model']}**: "
                f"{m['passed']}/{m['total']} passed ({m['accuracy']*100:.1f}%) | "
                f"RTF: {m['avg_rtf']:.3f} | Conf: {m['avg_confidence']:.3f}"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## Performance by Speaker",
            "",
            "| Speaker | Accent | Total | Passed | Failed | Accuracy |",
            "|---------|--------|-------|--------|--------|----------|"
        ])

        for s in stats["by_speaker"]:
            md_lines.append(
                f"| {s['speaker']} | {s['accent']} | {s['total']} | "
                f"{s['passed']} | {s['failed']} | {s['accuracy']*100:.2f}% |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## Performance by Accent",
            "",
            "| Accent | Total | Passed | Failed | Accuracy |",
            "|--------|-------|--------|--------|----------|"
        ])

        for a in stats["by_accent"]:
            md_lines.append(
                f"| {a['accent']} | {a['total']} | {a['passed']} | "
                f"{a['failed']} | {a['accuracy']*100:.2f}% |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## Performance by Model",
            "",
            "| Model | Total | Passed | Failed | Accuracy | Avg RTF | Avg Confidence |",
            "|-------|-------|--------|--------|----------|---------|----------------|"
        ])

        for m in stats["by_model"]:
            md_lines.append(
                f"| {m['model']} | {m['total']} | {m['passed']} | {m['failed']} | "
                f"{m['accuracy']*100:.2f}% | {m['avg_rtf']:.3f} | {m['avg_confidence']:.3f} |"
            )

        if stats["error_distribution"]:
            md_lines.extend([
                "",
                "---",
                "",
                "## Error Type Distribution",
                "",
                "| Error Type | Count |",
                "|------------|-------|"
            ])
            for err_type, count in sorted(stats["error_distribution"].items(), key=lambda x: x[1], reverse=True):
                md_lines.append(f"| {err_type} | {count} |")

        md_lines.extend([
            "",
            "---",
            "",
            "## Visualizations",
            "",
            "The following visualizations have been generated in the `visualizations/` directory:",
            "",
            "- **accent_model_heatmap.png**: Heatmap showing accuracy across accent × model combinations",
            "- **error_categorization.png**: Distribution of error types (substitution, deletion, insertion, ordering)",
            "- **speaker_breakdown.png**: Per-speaker performance comparison",
            "- **accent_correlation.png**: Correlation between exact match accuracy and accents",
            "- **per_condition_breakdown.png**: Performance breakdown by multiple conditions",
            ""
        ])

        summary_path = self.output_dir / "run_summary.md"
        summary_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info(f"Enhanced summary written to {summary_path}")

    def _generate_accent_model_heatmap(self, df: pd.DataFrame) -> str:
        """Generate heatmap of accent × model performance."""
        try:
            # Pivot table: rows=accents, cols=models, values=accuracy
            pivot_data = df.pivot_table(
                index="accent",
                columns=["model_name", "model_size"],
                values="numeric_exact_match",
                aggfunc="mean"
            )

            # Flatten column multi-index
            pivot_data.columns = [f"{m}/{s}" for m, s in pivot_data.columns]

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="RdYlGn",
                       vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Accuracy'})
            ax.set_title("Exact Match Accuracy: Accent × Model", fontsize=14, fontweight='bold')
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel("Accent", fontsize=12)
            plt.tight_layout()

            output_path = self.viz_dir / "accent_model_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(output_path)
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return ""

    def _generate_error_categorization(self, df: pd.DataFrame) -> str:
        """Generate error type categorization visualization."""
        try:
            if "error_type" not in df.columns:
                logger.warning("error_type column not found; skipping error categorization")
                return ""

            error_counts = df[df["numeric_exact_match"] == 0]["error_type"].value_counts()

            # Check if there are any errors to visualize
            if len(error_counts) == 0 or error_counts.sum() == 0:
                logger.info("No errors found; skipping error categorization visualization")
                return ""

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Bar chart
            error_counts.plot(kind='bar', ax=ax1, color=sns.color_palette("Set2"))
            ax1.set_title("Error Type Distribution", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Error Type", fontsize=12)
            ax1.set_ylabel("Count", fontsize=12)
            ax1.tick_params(axis='x', rotation=45)

            # Pie chart
            ax2.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%',
                   colors=sns.color_palette("Set2"), startangle=90)
            ax2.set_title("Error Type Proportion", fontsize=14, fontweight='bold')

            plt.tight_layout()

            output_path = self.viz_dir / "error_categorization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(output_path)
        except Exception as e:
            logger.error(f"Error categorization failed: {e}")
            return ""

    def _generate_speaker_breakdown(self, df: pd.DataFrame) -> str:
        """Generate per-speaker performance breakdown."""
        try:
            speaker_acc = df.groupby("speaker_id")["numeric_exact_match"].agg(['mean', 'count'])
            speaker_acc = speaker_acc.sort_values('mean', ascending=True)

            fig, ax = plt.subplots(figsize=(10, max(6, len(speaker_acc) * 0.3)))

            y_pos = np.arange(len(speaker_acc))
            bars = ax.barh(y_pos, speaker_acc['mean'], color=sns.color_palette("viridis", len(speaker_acc)))

            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{spk} (n={int(speaker_acc.loc[spk, 'count'])})"
                                for spk in speaker_acc.index])
            ax.set_xlabel("Accuracy", fontsize=12)
            ax.set_ylabel("Speaker", fontsize=12)
            ax.set_title("Per-Speaker Accuracy", fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)

            # Add accuracy labels on bars
            for i, (idx, row) in enumerate(speaker_acc.iterrows()):
                ax.text(row['mean'] + 0.02, i, f"{row['mean']*100:.1f}%",
                       va='center', fontsize=9)

            plt.tight_layout()

            output_path = self.viz_dir / "speaker_breakdown.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(output_path)
        except Exception as e:
            logger.error(f"Speaker breakdown failed: {e}")
            return ""

    def _generate_accent_correlation(self, df: pd.DataFrame) -> str:
        """Generate exact match correlation with accents."""
        try:
            accent_stats = df.groupby("accent").agg({
                "numeric_exact_match": ["mean", "count", "std"]
            })["numeric_exact_match"]
            accent_stats.columns = ["accuracy", "count", "std"]
            accent_stats = accent_stats.sort_values("accuracy", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 6))

            x_pos = np.arange(len(accent_stats))
            bars = ax.bar(x_pos, accent_stats["accuracy"],
                         yerr=accent_stats["std"],
                         capsize=5,
                         color=sns.color_palette("coolwarm", len(accent_stats)),
                         edgecolor='black', linewidth=0.7)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{acc}\n(n={int(accent_stats.loc[acc, 'count'])})"
                                for acc in accent_stats.index], rotation=45, ha='right')
            ax.set_ylabel("Exact Match Accuracy", fontsize=12)
            ax.set_xlabel("Accent", fontsize=12)
            ax.set_title("Exact Match Accuracy by Accent (with std dev)",
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=accent_stats["accuracy"].mean(), color='red',
                      linestyle='--', label=f'Mean: {accent_stats["accuracy"].mean():.2%}')
            ax.legend()

            plt.tight_layout()

            output_path = self.viz_dir / "accent_correlation.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(output_path)
        except Exception as e:
            logger.error(f"Accent correlation failed: {e}")
            return ""

    def _generate_per_condition_breakdown(self, df: pd.DataFrame) -> str:
        """Generate comprehensive per-condition breakdown heatmap."""
        try:
            # Create multi-dimensional analysis
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Accent × Noise Type
            if "noise_type" in df.columns:
                pivot1 = df.pivot_table(
                    index="accent",
                    columns="noise_type",
                    values="numeric_exact_match",
                    aggfunc="mean"
                )
                sns.heatmap(pivot1, annot=True, fmt=".2%", cmap="RdYlGn",
                           vmin=0, vmax=1, ax=axes[0, 0], cbar_kws={'label': 'Accuracy'})
                axes[0, 0].set_title("Accent × Noise Type", fontweight='bold')

            # 2. Speaker × Model
            pivot2 = df.groupby(["speaker_id", "model_name"])["numeric_exact_match"].mean().unstack()
            sns.heatmap(pivot2, annot=True, fmt=".2%", cmap="RdYlGn",
                       vmin=0, vmax=1, ax=axes[0, 1], cbar_kws={'label': 'Accuracy'})
            axes[0, 1].set_title("Speaker × Model", fontweight='bold')

            # 3. Number Range × Accent
            if "number_range" in df.columns:
                pivot3 = df.pivot_table(
                    index="number_range",
                    columns="accent",
                    values="numeric_exact_match",
                    aggfunc="mean"
                )
                sns.heatmap(pivot3, annot=True, fmt=".2%", cmap="RdYlGn",
                           vmin=0, vmax=1, ax=axes[1, 0], cbar_kws={'label': 'Accuracy'})
                axes[1, 0].set_title("Number Range × Accent", fontweight='bold')

            # 4. Model × Number Range
            if "number_range" in df.columns:
                pivot4 = df.pivot_table(
                    index="model_name",
                    columns="number_range",
                    values="numeric_exact_match",
                    aggfunc="mean"
                )
                sns.heatmap(pivot4, annot=True, fmt=".2%", cmap="RdYlGn",
                           vmin=0, vmax=1, ax=axes[1, 1], cbar_kws={'label': 'Accuracy'})
                axes[1, 1].set_title("Model × Number Range", fontweight='bold')

            plt.suptitle("Multi-Dimensional Performance Breakdown",
                        fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()

            output_path = self.viz_dir / "per_condition_breakdown.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(output_path)
        except Exception as e:
            logger.error(f"Per-condition breakdown failed: {e}")
            return ""

