#!/usr/bin/env python3
"""
Urban Accessibility Indicators Analysis

This script calculates key urban accessibility and structure indicators using geospatial data 
for the city of Queretaro, Mexico. 

Indicators calculated for each hexagonal grid cell:
- Total Population: Number of people residing in the cell
- Intersection Density: Number of street intersections per square kilometer
- Network Density: Total length of pedestrian network per square kilometer  
- Distance to Nearest Amenity: Shortest network distance to nearest food amenity

Author: Francisco Benita
Year: 2025
"""

import requests
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.ops import nearest_points
import contextily as cx
from tqdm import tqdm
import pickle


def download_data():
    """Download geospatial data files if they don't already exist."""
    print("Step 1: Downloading geospatial data...")
    
    # Dictionary of data names and their download URLs
    urls = {
        "boundary": "https://www.dropbox.com/scl/fi/h5oo7jrfkg146i9urm0i7/Queretaro_ZM.gpkg?rlkey=1hs3pn4me5mnqveg8xdlhfutx&st=qhccuqv2&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/fal6u5a1jb7igr0x012e2/queretaro_pedestrian_network.gpkg?rlkey=3zgofsbc0oj2a03l9nqc2iq96&st=h1bx74hs&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/t0qa7zijv60dad1rodqag/ZM_queretaro_pop_count.gpkg?rlkey=eheg5lt9j2aguuf4ofvzhj4l3&st=fsrvgpb7&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/rbfyiq7f00dvpccca7r4m/ZM_queretaro_poi_food.gpkg?rlkey=4a1ne13mruvgb64p7onoele0j&st=yhqlku75&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/fkhlsb226d1x9spirc47l/Queretaro_ZM_grid.gpkg?rlkey=eemz0bxofob6r62n23x411kvc&st=4we64nw7&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/w48l1qtoe3yyfzhsf7lq6/Queretaro_ZM_centroids.gpkg?rlkey=hx93d23vigyfnb0p4xy83y416&st=6mwlmm27&dl=1"
    }

    # Download each file if it doesn't exist
    for name, url in urls.items():
        filename = f"{name}.gpkg"
        
        if os.path.exists(filename):
            print(f"⏭️  File {filename} already exists, skipping download")
            continue
        
        try:
            print(f"⬇️  Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Successfully downloaded {filename}")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")


def load_and_visualize_data():
    """Load geospatial data and create initial visualization."""
    print("\nStep 2: Loading and visualizing data...")
    
    # Load all data layers
    boundary = gpd.read_file("boundary.gpkg")
    pedestrian_edges = gpd.read_file("pedestrian_network.gpkg", layer='edges')
    pedestrian_nodes = gpd.read_file("pedestrian_network.gpkg", layer='nodes')
    population = gpd.read_file("population_count.gpkg")
    poi_food = gpd.read_file("poi_food.gpkg")
    grid = gpd.read_file("grid.gpkg")
    centroids = gpd.read_file("centroids.gpkg")

    # Print CRS information
    print(f"Boundary CRS: {boundary.crs}")
    print(f"Grid CRS: {grid.crs}")

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot layers
    boundary.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    grid.plot(ax=ax, color='none', edgecolor='blue', linewidth=0.3, alpha=0.6)
    pedestrian_edges.plot(ax=ax, color='green', linewidth=0.5, alpha=0.7)
    poi_food.plot(ax=ax, color='red', markersize=10, alpha=0.8)
    population.plot(ax=ax, color='orange', markersize=2, alpha=0.6)
    centroids.plot(ax=ax, color='purple', markersize=1, alpha=0.8)

    # Add basemap
    try:
        cx.add_basemap(ax, crs=boundary.crs, source=cx.providers.CartoDB.Positron, alpha=0.8)
    except:
        print("Could not add basemap - continuing without it")

    ax.set_title('Queretaro Urban Analysis Layers Overview', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='lightgray', lw=4, label='City Boundary'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Hexagonal Grid'),
        plt.Line2D([0], [0], color='green', lw=2, label='Pedestrian Network'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Food POIs'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=6, label='Population'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=4, label='Grid Centroids')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Data overview plot saved as 'data_overview.png'")
    
    return boundary, pedestrian_edges, pedestrian_nodes, population, poi_food, grid, centroids


def calculate_population_indicator(grid, population):
    """Calculate total population for each hexagonal grid cell."""
    print("\nStep 3: Calculating population indicator...")
    
    # Check what columns are available in population data
    print(f"Population data columns: {list(population.columns)}")
    
    # Perform spatial join to count population points within each grid cell
    population_in_grid = gpd.sjoin(population, grid, how='inner', predicate='within')
    
    # Use the known population column name
    pop_column = 'Counts'
    print(f"Using population column: {pop_column}")
    
    # Group by grid cell ID and sum population
    pop_summary = population_in_grid.groupby('id')[pop_column].sum().reset_index()
    pop_summary.rename(columns={pop_column: 'total_population'}, inplace=True)
    
    # Merge back to grid
    grid = grid.merge(pop_summary, on='id', how='left')
    grid['total_population'].fillna(0, inplace=True)
    
    print(f"Population indicator calculated for {len(grid)} grid cells")
    return grid


def calculate_intersection_density(grid, pedestrian_nodes):
    """Calculate intersection density for each hexagonal grid cell."""
    print("\nStep 4: Calculating intersection density...")
    
    # Perform spatial join to count nodes (intersections) within each grid cell
    nodes_in_grid = gpd.sjoin(pedestrian_nodes, grid, how='inner', predicate='within')
    
    # Count intersections per grid cell
    intersection_counts = nodes_in_grid.groupby('id').size().reset_index(name='intersection_count')
    
    # Calculate area of each grid cell in square kilometers
    grid['area_km2'] = grid.geometry.area / 1_000_000
    
    # Merge intersection counts with grid
    grid = grid.merge(intersection_counts, on='id', how='left')
    grid['intersection_count'].fillna(0, inplace=True)
    
    # Calculate intersection density (intersections per km²)
    grid['intersection_density'] = grid['intersection_count'] / grid['area_km2']
    
    print(f"Intersection density calculated for {len(grid)} grid cells")
    return grid


def calculate_network_density(grid, pedestrian_edges):
    """Calculate network density for each hexagonal grid cell."""
    print("\nStep 5: Calculating network density...")
    
    # Clip pedestrian edges to each grid cell and sum lengths
    network_densities = []
    
    # Use tqdm to show progress
    for idx, cell in tqdm(grid.iterrows(), total=len(grid), desc="Processing grid cells"):
        try:
            # Clip edges to the current grid cell
            clipped_edges = gpd.clip(pedestrian_edges, cell.geometry)
            
            if not clipped_edges.empty:
                # Calculate total length of clipped edges
                total_length = clipped_edges['length'].sum()
                # Convert to density (meters per km²)
                density = total_length / (cell.geometry.area / 1_000_000)
            else:
                density = 0
                
            network_densities.append(density)
            
        except Exception as e:
            tqdm.write(f"Error processing cell {cell['id']}: {e}")
            network_densities.append(0)
    
    grid['network_density'] = network_densities
    
    print(f"Network density calculated for {len(grid)} grid cells")
    return grid


def get_nearest_node_with_distance(point, node_geometries, max_distance=500):
    """
    Find the nearest node to a point, but only if it's within max_distance meters.
    Returns (node_id, distance) or (None, None) if no node is within range.
    """
    # Find the geometry of the nearest node
    nearest_node_geom = nearest_points(point, node_geometries.union_all())[1]
    # Calculate distance to nearest node
    distance = point.distance(nearest_node_geom)
    
    # If distance is too large, exclude this point from analysis
    if distance > max_distance:
        return None, distance
        
    # Match the geometry back to the node ID (osmid)
    nearest_node_id = node_geometries[node_geometries == nearest_node_geom].index[0]
    return nearest_node_id, distance


def calculate_min_distance(start_node, target_nodes, graph):
    """Calculate minimum distance from start_node to any target_node."""
    try:
        # Find shortest path from start_node to any node in target_nodes
        dist = nx.multi_source_dijkstra(graph, sources=target_nodes, target=start_node, weight='length')[0]
        return dist
    except nx.NetworkXNoPath:
        # If no path exists, return infinity
        return float('inf')


def calculate_distance_to_amenity(grid, centroids, pedestrian_edges, pedestrian_nodes, poi_food):
    """Calculate distance to nearest amenity for each hexagonal grid cell."""
    print("\nStep 6: Calculating distance to nearest amenity...")
    
    # Check if cached results exist
    cache_file = "distance_to_amenity_cache.pkl"
    if os.path.exists(cache_file):
        print("Found cached distance calculations, loading...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Merge cached results with grid
            grid = grid.merge(cached_data[['id', 'dist_to_amenity']], on='id', how='left')
            print("✅ Cached distance data loaded successfully!")
            return grid
            
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            print("Proceeding with fresh calculation...")
    
    # Create network graph
    G = nx.from_pandas_edgelist(pedestrian_edges, 'u', 'v', ['length'])
    
    # Create spatial index for faster lookups
    node_geometries = pedestrian_nodes.set_index('osmid')['geometry']
    
    # Find nearest nodes for centroids and filter out those too far from network
    print("Finding nearest nodes and filtering hexagons...")
    
    # Use tqdm for finding nearest nodes
    tqdm.pandas(desc="Finding nearest nodes")
    centroid_results = centroids['geometry'].progress_apply(lambda p: get_nearest_node_with_distance(p, node_geometries))
    centroids['nearest_node'] = centroid_results.apply(lambda x: x[0])
    centroids['distance_to_network'] = centroid_results.apply(lambda x: x[1])
    
    # Filter out centroids that don't have nearby network nodes
    centroids_with_network = centroids[centroids['nearest_node'].notna()].copy()
    excluded_count = len(centroids) - len(centroids_with_network)
    
    print(f"Excluded {excluded_count} hexagons that are too far from pedestrian network")
    print(f"Analyzing {len(centroids_with_network)} hexagons with nearby network access")
    
    # Process POIs (use larger distance tolerance)
    print("Finding nearest nodes for POIs...")
    tqdm.pandas(desc="Processing POIs")
    poi_food['nearest_node'] = poi_food['geometry'].progress_apply(
        lambda p: get_nearest_node_with_distance(p, node_geometries, max_distance=1000)[0]
    )
    
    # Get unique POI nodes
    poi_nodes = set(poi_food['nearest_node'].dropna())
    
    # Calculate shortest path distances
    print("Calculating shortest path distances...")
    print(f"Processing {len(centroids_with_network)} hexagons instead of {len(centroids)}")
    
    # Use tqdm to show progress for distance calculations
    tqdm.pandas(desc="Calculating distances")
    centroids_with_network['dist_to_amenity'] = centroids_with_network['nearest_node'].progress_apply(
        lambda start_node: calculate_min_distance(start_node, poi_nodes, G)
    )
    
    # Merge results back
    centroids['dist_to_amenity'] = None
    centroids.loc[centroids_with_network.index, 'dist_to_amenity'] = centroids_with_network['dist_to_amenity']
    
    # Merge with grid
    grid = grid.merge(centroids[['id', 'dist_to_amenity']], on='id', how='left')
    
    # Save results to cache for future runs
    print("Saving distance calculations to cache...")
    try:
        cache_data = grid[['id', 'dist_to_amenity']].copy()
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("✅ Distance calculations cached successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not save cache: {e}")
    
    print("Distance to amenity calculation completed")
    return grid


def visualize_indicators(grid):
    """Create visualizations for all calculated indicators."""
    print("\nStep 7: Creating visualizations...")
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Urban Accessibility Indicators - Queretaro', fontsize=16, fontweight='bold')
    
    # Population
    grid.plot(column='total_population', ax=axes[0,0], legend=True, 
              cmap='YlOrRd', edgecolor='white', linewidth=0.1)
    axes[0,0].set_title('Total Population per Hexagon')
    axes[0,0].axis('off')
    
    # Intersection Density
    grid.plot(column='intersection_density', ax=axes[0,1], legend=True,
              cmap='Blues', edgecolor='white', linewidth=0.1)
    axes[0,1].set_title('Intersection Density (per km²)')
    axes[0,1].axis('off')
    
    # Network Density
    grid.plot(column='network_density', ax=axes[1,0], legend=True,
              cmap='Greens', edgecolor='white', linewidth=0.1)
    axes[1,0].set_title('Network Density (m/km²)')
    axes[1,0].axis('off')
    
    # Distance to Amenity
    grid_with_amenity = grid[grid['dist_to_amenity'].notna()]
    grid_with_amenity.plot(column='dist_to_amenity', ax=axes[1,1], legend=True,
                          cmap='Reds_r', edgecolor='white', linewidth=0.1)
    axes[1,1].set_title('Distance to Nearest Food Amenity (m)')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('urban_indicators.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Urban indicators plot saved as 'urban_indicators.png'")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total Population: {grid['total_population'].sum():,.0f}")
    print(f"Average Intersection Density: {grid['intersection_density'].mean():.2f} per km²")
    print(f"Average Network Density: {grid['network_density'].mean():.0f} m/km²")
    print(f"Average Distance to Food Amenity: {grid['dist_to_amenity'].mean():.0f} m")
    print(f"Hexagons with network access: {grid['dist_to_amenity'].notna().sum()}/{len(grid)}")


def main():
    """Main execution function."""
    print("Urban Accessibility Indicators Analysis")
    print("=====================================")
    
    # Step 1: Download data
    download_data()
    
    # Step 2: Load and visualize data
    boundary, pedestrian_edges, pedestrian_nodes, population, poi_food, grid, centroids = load_and_visualize_data()
    
    # Step 3: Calculate indicators
    grid = calculate_population_indicator(grid, population)
    grid = calculate_intersection_density(grid, pedestrian_nodes)
    grid = calculate_network_density(grid, pedestrian_edges)
    grid = calculate_distance_to_amenity(grid, centroids, pedestrian_edges, pedestrian_nodes, poi_food)
    
    # Step 4: Visualize results
    visualize_indicators(grid)
    
    # Save final processed grid with all indicators
    print("\nSaving final results...")
    try:
        grid.to_file("processed_grid_results.gpkg", driver="GPKG")
        print("✅ Final grid results saved to 'processed_grid_results.gpkg'")
    except Exception as e:
        print(f"⚠️  Warning: Could not save final results: {e}")
    
    print("\n✅ Analysis completed successfully!")
    return grid


if __name__ == "__main__":
    result_grid = main()