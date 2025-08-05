import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob

# Set publication-ready style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3
})

def load_and_process_data(data_directory=None, file_pattern="*_*_*.json"):
    """
    Load all JSON files matching the pattern and process them into a DataFrame
    
    Parameters:
    data_directory (str): Path to directory containing JSON files (if None, uses current directory)
    file_pattern (str): Pattern to match JSON files
    
    Returns:
    pd.DataFrame: Processed data with microscope, species, genotype columns
    """
    all_data = []
    
    # Construct full path pattern
    if data_directory is not None:
        full_pattern = str(Path(data_directory) / file_pattern)
    else:
        full_pattern = file_pattern
    
    # Find all matching JSON files
    json_files = glob.glob(full_pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {file_pattern}")
        print("Please make sure your JSON files are in the current directory")
        return pd.DataFrame()
    
    for file_path in json_files:
        # Parse filename: Microscope_Species_Genotype.json
        filename = Path(file_path).stem
        parts = filename.split('_')
        
        if len(parts) >= 3:
            microscope = parts[0]
            species = parts[1]
            genotype = '_'.join(parts[2:])  # Handle genotypes with underscores
        else:
            print(f"Warning: Unexpected filename format: {filename}")
            continue
        
        # Load JSON data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame and add metadata
            df = pd.DataFrame(data)
            df['microscope'] = microscope
            df['species'] = species
            df['genotype'] = genotype
            df['combination'] = f"{microscope}_{species}"
            
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {filename}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal samples loaded: {len(combined_df)}")
        print(f"Microscope-Species combinations: {combined_df['combination'].unique()}")
        return combined_df
    else:
        return pd.DataFrame()

def create_publication_plots(df, metrics=['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']):
    """
    Create publication-ready plots for segmentation metrics
    
    Parameters:
    df (pd.DataFrame): Processed data
    metrics (list): List of metrics to plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Calculate number of rows and columns for subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure with subplots - increased height for more padding
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes = axes.flatten()
    
    # Color palette
    combinations = sorted(df['combination'].unique())
    colors = sns.color_palette("husl", len(combinations))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create violin plot with box plot overlay
        violin_parts = ax.violinplot([df[df['combination'] == comb][metric].values 
                                    for comb in combinations],
                                   positions=range(len(combinations)),
                                   showmeans=True, showmedians=True, showextrema=False)
        
        # Customize violin plot colors
        for j, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        
        # Add box plots for better statistical visualization
        bp = ax.boxplot([df[df['combination'] == comb][metric].values 
                        for comb in combinations],
                       positions=range(len(combinations)),
                       patch_artist=True, widths=0.3,
                       boxprops=dict(facecolor='white', alpha=0.8),
                       medianprops=dict(color='black', linewidth=2))
        
        # Customize appearance
        ax.set_xlabel('Microscope-Species Combination', fontweight='bold')
        ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
        ax.set_title(f'Distribution of {metric.upper()} Scores', fontweight='bold', pad=40)  # Increased padding
        
        # Set x-axis labels
        ax.set_xticks(range(len(combinations)))
        ax.set_xticklabels(combinations, rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add sample size annotations - position them higher above the plot
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        y_range = y_max - y_min
        
        # Position annotations well above the highest data point
        annotation_height = y_max + (y_range * 0.08)  # Position above the plot area
        
        for j, comb in enumerate(combinations):
            n_samples = len(df[df['combination'] == comb])
            ax.text(j, annotation_height, f'n={n_samples}', 
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Extend y-axis to accommodate annotations with extra space
        ax.set_ylim(y_min, y_max + (y_range * 0.15))  # More space for annotations
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout with much more spacing
    plt.tight_layout(pad=4.0, h_pad=6.0, w_pad=3.0)
    
    # Save figures with extra padding
    plt.savefig('segmentation_metrics_violin_plots.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.savefig('segmentation_metrics_violin_plots.pdf', bbox_inches='tight', pad_inches=0.3)
    plt.show()

def create_genotype_comparison_plots(df, microscope=None, species=None, metrics=['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']):
    """
    Create plots comparing different genotypes for a specific microscope-species combination
    
    Parameters:
    df (pd.DataFrame): Processed data
    microscope (str): Specific microscope to analyze (if None, will show available options)
    species (str): Specific species to analyze (if None, will show available options)
    metrics (list): List of metrics to plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Show available combinations if not specified
    if microscope is None or species is None:
        print("Available microscope-species combinations:")
        combinations = df['combination'].unique()
        for combo in sorted(combinations):
            mic, spec = combo.split('_', 1)
            print(f"  Microscope: {mic}, Species: {spec}")
        print("\nPlease specify both microscope and species parameters.")
        return
    
    # Filter data for specific microscope-species combination
    filtered_df = df[(df['microscope'] == microscope) & (df['species'] == species)]
    
    if filtered_df.empty:
        print(f"No data found for microscope '{microscope}' and species '{species}'")
        print("Available combinations:")
        combinations = df['combination'].unique()
        for combo in sorted(combinations):
            mic, spec = combo.split('_', 1)
            print(f"  Microscope: {mic}, Species: {spec}")
        return
    
    # Get unique genotypes
    genotypes = sorted(filtered_df['genotype'].unique())
    
    if len(genotypes) < 2:
        print(f"Only {len(genotypes)} genotype(s) found for {microscope}-{species}. Need at least 2 for comparison.")
        print(f"Available genotypes: {genotypes}")
        return
    
    print(f"Creating genotype comparison plots for {microscope}-{species}")
    print(f"Genotypes found: {genotypes}")
    
    # Calculate subplot layout
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes = axes.flatten()
    
    # Color palette for genotypes
    colors = sns.color_palette("Set2", len(genotypes))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create violin plot with box plot overlay
        data_by_genotype = [filtered_df[filtered_df['genotype'] == geno][metric].values 
                           for geno in genotypes]
        
        violin_parts = ax.violinplot(data_by_genotype,
                                   positions=range(len(genotypes)),
                                   showmeans=True, showmedians=True, showextrema=False)
        
        # Customize violin plot colors
        for j, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        
        # Add box plots
        bp = ax.boxplot(data_by_genotype,
                       positions=range(len(genotypes)),
                       patch_artist=True, widths=0.3,
                       boxprops=dict(facecolor='white', alpha=0.8),
                       medianprops=dict(color='black', linewidth=2))
        
        # Customize appearance
        ax.set_xlabel('Genotype', fontweight='bold')
        ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
        ax.set_title(f'{metric.upper()} Scores by Genotype\n({microscope} - {species})', 
                    fontweight='bold', pad=40)
        
        # Set x-axis labels
        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add sample size annotations - position them higher above the plot
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        y_range = y_max - y_min
        
        # Position annotations well above the highest data point
        annotation_height = y_max + (y_range * 0.08)
        
        for j, genotype in enumerate(genotypes):
            n_samples = len(filtered_df[filtered_df['genotype'] == genotype])
            ax.text(j, annotation_height, f'n={n_samples}', 
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Extend y-axis to accommodate annotations with extra space
        ax.set_ylim(y_min, y_max + (y_range * 0.15))
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout with much more spacing
    plt.tight_layout(pad=4.0, h_pad=6.0, w_pad=3.0)
    
    # Save figures with descriptive names
    filename_base = f'genotype_comparison_{microscope}_{species}'
    plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.savefig(f'{filename_base}.pdf', bbox_inches='tight', pad_inches=0.3)
    plt.show()
    
    # Print summary for this comparison
    print(f"\nGenotype comparison summary for {microscope}-{species}:")
    print("="*60)
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for genotype in genotypes:
            subset = filtered_df[filtered_df['genotype'] == genotype]
            mean_val = subset[metric].mean()
            std_val = subset[metric].std()
            n_samples = len(subset)
            print(f"  {genotype}: {mean_val:.4f} ± {std_val:.4f} (n={n_samples})")

def create_statistical_table_figure(df, table_type="microscope_species", microscope=None, species=None, 
                                   metrics=['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']):
    """
    Create publication-ready statistical summary table as a figure
    
    Parameters:
    df (pd.DataFrame): Processed data
    table_type (str): Type of table - "microscope_species" or "genotype"
    microscope (str): For genotype tables, specify microscope
    species (str): For genotype tables, specify species
    metrics (list): List of metrics to include
    """
    if df.empty:
        print("No data to create table")
        return
    
    # Prepare data based on table type
    if table_type == "microscope_species":
        # Table for microscope-species combinations
        combinations = sorted(df['combination'].unique())
        groups = combinations
        group_column = 'combination'
        title = "Statistical Summary: Microscope-Species Combinations"
        filename_base = "statistical_summary_microscope_species"
        
    elif table_type == "genotype":
        if microscope is None or species is None:
            print("For genotype tables, please specify both microscope and species")
            return
            
        # Filter for specific microscope-species combination
        filtered_df = df[(df['microscope'] == microscope) & (df['species'] == species)]
        if filtered_df.empty:
            print(f"No data found for {microscope}-{species}")
            return
            
        genotypes = sorted(filtered_df['genotype'].unique())
        groups = genotypes
        group_column = 'genotype'
        df = filtered_df  # Use filtered data
        title = f"Statistical Summary: Genotypes ({microscope} - {species})"
        filename_base = f"statistical_summary_genotype_{microscope}_{species}"
    
    # Create statistical summary data
    table_data = []
    
    for group in groups:
        if table_type == "microscope_species":
            subset = df[df[group_column] == group]
            group_name = group.replace('_', ' ')
        else:
            subset = df[df[group_column] == group]
            group_name = group
            
        n_samples = len(subset)
        
        row_data = {'Group': group_name, 'N': n_samples}
        
        for metric in metrics:
            mean_val = subset[metric].mean()
            std_val = subset[metric].std()
            row_data[f'{metric.upper()}_mean'] = mean_val
            row_data[f'{metric.upper()}_std'] = std_val
            row_data[f'{metric.upper()}'] = f"{mean_val:.3f} ± {std_val:.3f}"
        
        table_data.append(row_data)
    
    # Create DataFrame for the table
    table_df = pd.DataFrame(table_data)
    
    # Create figure for the table
    fig, ax = plt.subplots(figsize=(16, max(6, len(groups) * 0.6 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data for display (only mean±std columns)
    display_columns = ['Group', 'N'] + [f'{metric.upper()}' for metric in metrics]
    display_data = table_df[display_columns].values
    
    # Create table
    table = ax.table(cellText=display_data,
                    colLabels=display_columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Style header row
    for i in range(len(display_columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Style data rows - alternating colors
    for i in range(1, len(groups) + 1):
        row_color = '#F2F2F2' if i % 2 == 0 else 'white'
        for j in range(len(display_columns)):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_height(0.06)
            
            # Bold the group names
            if j == 0:  # Group column
                table[(i, j)].set_text_props(weight='bold')
    
    # Add title
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Add subtitle with explanation
    subtitle = "Values shown as Mean ± Standard Deviation"
    plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.1)
    
    # Save table figure
    plt.savefig(f'{filename_base}_table.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(f'{filename_base}_table.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()
    
    # Also save as CSV for reference
    csv_filename = f'{filename_base}_data.csv'
    
    # Prepare detailed CSV with separate mean and std columns
    csv_data = []
    for group in groups:
        if table_type == "microscope_species":
            subset = df[df[group_column] == group]
            group_name = group.replace('_', ' ')
        else:
            subset = df[df[group_column] == group]
            group_name = group
            
        n_samples = len(subset)
        row_data = {'Group': group_name, 'N': n_samples}
        
        for metric in metrics:
            mean_val = subset[metric].mean()
            std_val = subset[metric].std()
            min_val = subset[metric].min()
            max_val = subset[metric].max()
            median_val = subset[metric].median()
            
            row_data[f'{metric.upper()}_Mean'] = round(mean_val, 4)
            row_data[f'{metric.upper()}_Std'] = round(std_val, 4)
            row_data[f'{metric.upper()}_Min'] = round(min_val, 4)
            row_data[f'{metric.upper()}_Max'] = round(max_val, 4)
            row_data[f'{metric.upper()}_Median'] = round(median_val, 4)
        
        csv_data.append(row_data)
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_filename, index=False)
    
    print(f"Statistical table created and saved as:")
    print(f"  - {filename_base}_table.png (figure)")
    print(f"  - {filename_base}_table.pdf (figure)")
    print(f"  - {csv_filename} (detailed data)")
    
    return table_df
    """
    Create genotype comparison plots for all available microscope-species combinations
    
    Parameters:
    df (pd.DataFrame): Processed data
    metrics (list): List of metrics to plot
    """
    if df.empty:
        print("No data available")
        return
    
    # Get all unique microscope-species combinations
    combinations = df['combination'].unique()
    
    print("Creating genotype comparison plots for all combinations...")
    print("="*60)
    
    for combination in sorted(combinations):
        microscope, species = combination.split('_', 1)
        
        # Check if this combination has multiple genotypes
        subset = df[df['combination'] == combination]
        genotypes = subset['genotype'].unique()
        
        if len(genotypes) >= 2:
            print(f"\nProcessing {microscope}-{species} ({len(genotypes)} genotypes)...")
            create_genotype_comparison_plots(df, microscope, species, metrics)
            
            # Create statistical table for this genotype comparison
            print(f"Creating statistical table for {microscope}-{species} genotypes...")
            create_statistical_table_figure(df, table_type="genotype", 
                                           microscope=microscope, species=species, metrics=metrics)
        else:
            print(f"\nSkipping {microscope}-{species} (only {len(genotypes)} genotype: {genotypes[0]})")

def create_statistical_table_figure(df, table_type="microscope_species", microscope=None, species=None, 
                                   metrics=['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']):
    """
    Create publication-ready statistical summary table as a figure
    
    Parameters:
    df (pd.DataFrame): Processed data
    table_type (str): Type of table - "microscope_species" or "genotype"
    microscope (str): For genotype tables, specify microscope
    species (str): For genotype tables, specify species
    metrics (list): List of metrics to include
    """
    if df.empty:
        print("No data to create table")
        return
    
    # Prepare data based on table type
    if table_type == "microscope_species":
        # Table for microscope-species combinations
        combinations = sorted(df['combination'].unique())
        groups = combinations
        group_column = 'combination'
        title = "Statistical Summary: Microscope-Species Combinations"
        filename_base = "statistical_summary_microscope_species"
        
    elif table_type == "genotype":
        if microscope is None or species is None:
            print("For genotype tables, please specify both microscope and species")
            return
            
        # Filter for specific microscope-species combination
        filtered_df = df[(df['microscope'] == microscope) & (df['species'] == species)]
        if filtered_df.empty:
            print(f"No data found for {microscope}-{species}")
            return
            
        genotypes = sorted(filtered_df['genotype'].unique())
        groups = genotypes
        group_column = 'genotype'
        df = filtered_df  # Use filtered data
        title = f"Statistical Summary: Genotypes ({microscope} - {species})"
        filename_base = f"statistical_summary_genotype_{microscope}_{species}"
    
    # Create statistical summary data
    table_data = []
    
    for group in groups:
        if table_type == "microscope_species":
            subset = df[df[group_column] == group]
            group_name = group.replace('_', ' ')
        else:
            subset = df[df[group_column] == group]
            group_name = group
            
        n_samples = len(subset)
        
        row_data = {'Group': group_name, 'N': n_samples}
        
        for metric in metrics:
            mean_val = subset[metric].mean()
            std_val = subset[metric].std()
            row_data[f'{metric.upper()}_mean'] = mean_val
            row_data[f'{metric.upper()}_std'] = std_val
            row_data[f'{metric.upper()}'] = f"{mean_val:.3f} ± {std_val:.3f}"
        
        table_data.append(row_data)
    
    # Create DataFrame for the table
    table_df = pd.DataFrame(table_data)
    
    # Create figure for the table
    fig, ax = plt.subplots(figsize=(16, max(6, len(groups) * 0.6 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data for display (only mean±std columns)
    display_columns = ['Group', 'N'] + [f'{metric.upper()}' for metric in metrics]
    display_data = table_df[display_columns].values
    
    # Create table
    table = ax.table(cellText=display_data,
                    colLabels=display_columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.2, 2.0)
    
    # Style header row
    for i in range(len(display_columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Style data rows - alternating colors
    for i in range(1, len(groups) + 1):
        row_color = '#F2F2F2' if i % 2 == 0 else 'white'
        for j in range(len(display_columns)):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_height(0.06)
            
            # Bold the group names
            if j == 0:  # Group column
                table[(i, j)].set_text_props(weight='bold')
    
    # Add title
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Add subtitle with explanation
    subtitle = "Values shown as Mean ± Standard Deviation"
    plt.figtext(0.5, 0.02, subtitle, ha='center', fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.1)
    
    # Save table figure
    plt.savefig(f'{filename_base}_table.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(f'{filename_base}_table.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()
    
    # Also save as CSV for reference
    csv_filename = f'{filename_base}_data.csv'
    
    # Prepare detailed CSV with separate mean and std columns
    csv_data = []
    for group in groups:
        if table_type == "microscope_species":
            subset = df[df[group_column] == group]
            group_name = group.replace('_', ' ')
        else:
            subset = df[df[group_column] == group]
            group_name = group
            
        n_samples = len(subset)
        row_data = {'Group': group_name, 'N': n_samples}
        
        for metric in metrics:
            mean_val = subset[metric].mean()
            std_val = subset[metric].std()
            min_val = subset[metric].min()
            max_val = subset[metric].max()
            median_val = subset[metric].median()
            
            row_data[f'{metric.upper()}_Mean'] = round(mean_val, 4)
            row_data[f'{metric.upper()}_Std'] = round(std_val, 4)	
            row_data[f'{metric.upper()}_Min'] = round(min_val, 4)
            row_data[f'{metric.upper()}_Max'] = round(max_val, 4)
            row_data[f'{metric.upper()}_Median'] = round(median_val, 4)
        
        csv_data.append(row_data)
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_filename, index=False)
    
    print(f"Statistical table created and saved as:")
    print(f"  - {filename_base}_table.png (figure)")
    print(f"  - {filename_base}_table.pdf (figure)")
    print(f"  - {csv_filename} (detailed data)")
    
    return table_df

def create_summary_statistics(df):
    """
    Create summary statistics table
    
    Parameters:
    df (pd.DataFrame): Processed data
    """
    if df.empty:
        return
    
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']
    
    # Calculate summary statistics by combination
    summary_stats = []
    
    for combination in sorted(df['combination'].unique()):
        subset = df[df['combination'] == combination]
        
        for metric in metrics:
            stats = {
                'Combination': combination,
                'Metric': metric.upper(),
                'Count': len(subset),
                'Mean': subset[metric].mean(),
                'Std': subset[metric].std(),
                'Min': subset[metric].min(),
                'Q25': subset[metric].quantile(0.25),
                'Median': subset[metric].median(),
                'Q75': subset[metric].quantile(0.75),
                'Max': subset[metric].max()
            }
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Format numerical columns
    numerical_cols = ['Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max']
    for col in numerical_cols:
        summary_df[col] = summary_df[col].round(4)
    
    print("\nSummary Statistics:")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv('segmentation_metrics_summary.csv', index=False)
    print(f"\nSummary statistics saved to 'segmentation_metrics_summary.csv'")
    
    return summary_df

def main():
    """
    Main function to run the analysis
    """
    print("Loading and processing segmentation metrics data...")
    print("="*60)

    data_dir = r"C:\Users\Yifei\Documents\data_for_publication\results\inference_overlays\metrics"
    
    # Load data
    df = load_and_process_data(data_directory=data_dir, file_pattern="*_*_*.json")
    
    if df.empty:
        print("No data loaded. Please check your file pattern and ensure JSON files are present.")
        return
    
    # Create plots
    print("\nCreating publication-ready plots...")
    create_publication_plots(df)
    
    # Create statistical summary table for microscope-species combinations
    print("\nCreating statistical summary table for microscope-species combinations...")
    create_statistical_table_figure(df, table_type="microscope_species")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    create_summary_statistics(df)
    
    # Create genotype comparison plots
    print("\n" + "="*60)
    print("GENOTYPE COMPARISON ANALYSIS")
    print("="*60)
    
    # # Option 1: Create plots for all combinations automatically
    # create_all_genotype_comparisons(df)
    
    # Option 2: Create plot for specific combination (uncomment and modify as needed)
    # Example usage:
    # create_genotype_comparison_plots(df, microscope='Olympus', species='Rice')
    
    print("\nTo create a specific genotype comparison plot, use:")
    print("create_genotype_comparison_plots(df, microscope='YourMicroscope', species='YourSpecies')")
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- segmentation_metrics_violin_plots.png/pdf (microscope-species comparison)")
    print("- statistical_summary_microscope_species_table.png/pdf (summary table)")
    print("- genotype_comparison_[microscope]_[species].png/pdf (genotype comparisons)")
    print("- statistical_summary_genotype_[microscope]_[species]_table.png/pdf (genotype tables)")
    print("- segmentation_metrics_summary.csv (detailed summary statistics)")
    print("- statistical_summary_*_data.csv (table data files)")

if __name__ == "__main__":
    main()