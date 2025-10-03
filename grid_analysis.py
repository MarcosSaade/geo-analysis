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


def download_data(city_name, urls):
    """Download geospatial data files if they don't already exist."""
    print(f"Step 1: Downloading geospatial data for {city_name}...")
    
    # Create city-specific directory
    city_dir = f"data_{city_name}"
    os.makedirs(city_dir, exist_ok=True)
    
    # Download each file if it doesn't exist
    for name, url in urls.items():
        filename = os.path.join(city_dir, f"{name}.gpkg")
        
        if os.path.exists(filename):
            print(f"⏭️  File {name}.gpkg already exists, skipping download")
            continue
        
        try:
            print(f"⬇️  Downloading {name}.gpkg...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Successfully downloaded {name}.gpkg")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
    
    return city_dir


def load_and_visualize_data(city_name, city_dir):
    """Load geospatial data and create initial visualization."""
    print(f"\nStep 2: Loading and visualizing data for {city_name}...")
    
    # Load all data layers from city-specific directory
    boundary = gpd.read_file(os.path.join(city_dir, "boundary.gpkg"))
    pedestrian_edges = gpd.read_file(os.path.join(city_dir, "pedestrian_network.gpkg"), layer='edges')
    pedestrian_nodes = gpd.read_file(os.path.join(city_dir, "pedestrian_network.gpkg"), layer='nodes')
    population = gpd.read_file(os.path.join(city_dir, "population_count.gpkg"))
    poi_food = gpd.read_file(os.path.join(city_dir, "poi_food.gpkg"))
    grid = gpd.read_file(os.path.join(city_dir, "grid.gpkg"))
    centroids = gpd.read_file(os.path.join(city_dir, "centroids.gpkg"))

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
    output_file = os.path.join(city_dir, f'data_overview_{city_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data overview plot saved as '{output_file}'")
    
    return boundary, pedestrian_edges, pedestrian_nodes, population, poi_food, grid, centroids


def calculate_population_indicator(grid, population, city_name):
    """Calculate total population for each hexagonal grid cell."""
    print(f"\nStep 3: Calculating population indicator for {city_name}...")
    
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


def calculate_intersection_density(grid, pedestrian_nodes, city_name):
    """Calculate intersection density for each hexagonal grid cell."""
    print(f"\nStep 4: Calculating intersection density for {city_name}...")
    
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


def calculate_network_density(grid, pedestrian_edges, city_name):
    """Calculate network density for each hexagonal grid cell."""
    print(f"\nStep 5: Calculating network density for {city_name}...")
    
    # Clip pedestrian edges to each grid cell and sum lengths
    network_densities = []
    
    # Use tqdm to show progress
    for idx, cell in tqdm(grid.iterrows(), total=len(grid), desc=f"Processing {city_name} grid cells"):
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


def calculate_distance_to_amenity(grid, centroids, pedestrian_edges, pedestrian_nodes, poi_food, city_name, city_dir):
    """Calculate distance to nearest amenity for each hexagonal grid cell."""
    print(f"\nStep 6: Calculating distance to nearest amenity for {city_name}...")
    
    # Check if cached results exist
    cache_file = os.path.join(city_dir, "distance_to_amenity_cache.pkl")
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
    
    # Better filtering: Check which hexagons actually contain pedestrian network
    print("Filtering hexagons that contain pedestrian network...")
    
    # Find hexagons that actually intersect with pedestrian edges
    print("Checking spatial intersection with pedestrian network...")
    hexagons_with_network = []
    
    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc=f"Checking {city_name} network coverage"):
        # Check if this hexagon intersects with any pedestrian edges
        intersects = pedestrian_edges.intersects(row.geometry).any()
        if intersects:
            hexagons_with_network.append(row['id'])
    
    # Filter centroids to only those in hexagons with network
    centroids_with_network = centroids[centroids['id'].isin(hexagons_with_network)].copy()
    excluded_count = len(centroids) - len(centroids_with_network)
    
    print(f"Excluded {excluded_count} hexagons without pedestrian network")
    print(f"Analyzing {len(centroids_with_network)} hexagons with actual network coverage")
    
    # Find nearest nodes only for centroids in hexagons with network
    print("Finding nearest nodes for hexagons with network...")
    tqdm.pandas(desc="Finding nearest nodes")
    centroid_results = centroids_with_network['geometry'].progress_apply(lambda p: get_nearest_node_with_distance(p, node_geometries, max_distance=1000))
    centroids_with_network['nearest_node'] = centroid_results.apply(lambda x: x[0])
    centroids_with_network['distance_to_network'] = centroid_results.apply(lambda x: x[1])
    
    # Remove any that still don't have nearby nodes (shouldn't happen but safety check)
    centroids_with_network = centroids_with_network[centroids_with_network['nearest_node'].notna()].copy()
    final_excluded = len(centroids) - len(centroids_with_network)
    
    if final_excluded > excluded_count:
        print(f"Additional {final_excluded - excluded_count} hexagons excluded due to no nearby network nodes")
    
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
    
    # Merge results back - only for hexagons with network coverage
    centroids['dist_to_amenity'] = None
    # Use ID-based merging to ensure correct alignment
    network_results = centroids_with_network[['id', 'dist_to_amenity']].set_index('id')
    for idx, row in centroids.iterrows():
        if row['id'] in network_results.index:
            centroids.loc[idx, 'dist_to_amenity'] = network_results.loc[row['id'], 'dist_to_amenity']
    
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


def visualize_indicators(grid, city_name, city_dir):
    """Create visualizations for all calculated indicators."""
    print(f"\nStep 7: Creating visualizations for {city_name}...")
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Urban Accessibility Indicators - {city_name}', fontsize=16, fontweight='bold')
    
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
    output_file = os.path.join(city_dir, f'urban_indicators_{city_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Urban indicators plot saved as '{output_file}'")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS FOR {city_name.upper()} ===")
    print(f"Total Population: {grid['total_population'].sum():,.0f}")
    print(f"Average Intersection Density: {grid['intersection_density'].mean():.2f} per km²")
    print(f"Average Network Density: {grid['network_density'].mean():.0f} m/km²")
    print(f"Average Distance to Food Amenity: {grid['dist_to_amenity'].mean():.0f} m")
    print(f"Hexagons with network access: {grid['dist_to_amenity'].notna().sum()}/{len(grid)}")


def process_city(city_name, city_urls):
    """Process a single city - download data, calculate indicators, and visualize."""
    print(f"\n{'='*80}")
    print(f"PROCESSING CITY: {city_name.upper()}")
    print(f"{'='*80}\n")
    
    # Step 1: Download data
    city_dir = download_data(city_name, city_urls)
    
    # Step 2: Load and visualize data
    boundary, pedestrian_edges, pedestrian_nodes, population, poi_food, grid, centroids = load_and_visualize_data(city_name, city_dir)
    
    # Step 3: Calculate indicators
    grid = calculate_population_indicator(grid, population, city_name)
    grid = calculate_intersection_density(grid, pedestrian_nodes, city_name)
    grid = calculate_network_density(grid, pedestrian_edges, city_name)
    # grid = calculate_distance_to_amenity(grid, centroids, pedestrian_edges, pedestrian_nodes, poi_food, city_name, city_dir)
    
    # Step 4: Visualize results
    visualize_indicators(grid, city_name, city_dir)
    
    # Save final processed grid with all indicators
    print(f"\nSaving final results for {city_name}...")
    try:
        output_file = os.path.join(city_dir, f"processed_grid_results_{city_name}.gpkg")
        grid.to_file(output_file, driver="GPKG")
        print(f"✅ Final grid results saved to '{output_file}'")
    except Exception as e:
        print(f"⚠️  Warning: Could not save final results: {e}")
    
    print(f"\n✅ Analysis completed successfully for {city_name}!")
    return grid


def main():
    """Main execution function."""
    print("Urban Accessibility Indicators Analysis - Multi-City Processing")
    print("="*80)
    
    # Dictionary of cities and their data URLs
    urls = {
    "Ciudad de Mexico": {
        "boundary": "https://www.dropbox.com/scl/fi/ut8v6r54bf7votisjsinp/ZM_CDMX.gpkg?rlkey=cuvgo96rcddwgwbiwcqtbhgh8&st=uxf0grb9&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/4jehpo3wqhzsh5piu4fju/cdmx_pedestrian_network.gpkg?rlkey=y5a5qcogv30b8ei3xjmg0tg7d&st=repl2kw2&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/9zqpenj5of6dh3roe415d/ZM_CDMX_pop_count.gpkg?rlkey=pveese4i72mzux0xnu83d3emu&st=sg1395re&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/7jn6b25n7bw19a7t432da/ZM_CDMX_poi_food.gpkg?rlkey=zz6yn05b6cdhozltwk8y7h5un&st=hqkvqaqb&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/1l8vls67z508zbtyc59lh/CDMX_ZM_grid.gpkg?rlkey=15eg31udo3vwbxll3gtxgpeqh&st=6sjf6i1m&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/fuodav4nm53vvtrwrfmsa/CDMX_ZM_centroids.gpkg?rlkey=wt40sbusj1vs88cz6k431gn2h&st=c3o6o2sn&dl=1"
    },


    "Monterrey": {
        "boundary": "https://www.dropbox.com/scl/fi/np2nkbtkgat78bbr0s7dw/Monterrey_ZM.gpkg?rlkey=0jk3znfz0rk9tg5m6a8ikmfn0&st=2azh8xih&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/21kaaeudrojptosjhl1w6/monterrey_pedestrian_network.gpkg?rlkey=b1du48lfj1kx77xyykrisdifu&st=mr8ybkt9&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/jn4b2k2g7cfdjv8axzgwh/ZM_monterrey_pop_count.gpkg?rlkey=azp2vyh96msarizx8740l5k3e&st=628u61gl&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/51jyf0yqqw76te1halbki/ZM_monterrey_poi_food.gpkg?rlkey=xj4vn4mjixs05v30t0j9yk40o&st=kbbrvgd7&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/9l3azcn83xl4shwx6kyk3/Monterrey_ZM_grid.gpkg?rlkey=xidswz5jiptkstxenrxy4bmtu&st=x5tpdfaj&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/da4wa3m2zi156gid0l1hr/Monterrey_ZM_centroids.gpkg?rlkey=caq151syyznb1e1w8mnfhb20g&st=cv9ym66w&dl=1"
    },


    "Guadalajara": {
        "boundary": "https://www.dropbox.com/scl/fi/64c5zr15skwcjzeg6zize/Guadalajara_ZM.gpkg?rlkey=lbbof4a3zg2wk2qza62sepywj&st=zqnn7ter&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/nn6k3oqmtnlqhh495gf2g/guadalajara_pedestrian_network.gpkg?rlkey=evta4lq5sich8n4bxi32acq9m&st=7q6v5ch4&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/oeagwv2v3tb8bhnft5p58/ZM_guadalajara_pop_count.gpkg?rlkey=ksvf7q4f1kuzrrbma698jevaj&st=2wuyetz8&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/1lek1qup0pwm3zr86hn45/ZM_guadalajara_poi_food.gpkg?rlkey=a3s3ly0bh3gld2twcezb9fz4r&st=zv8ibce5&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/3o6sigukwuh7houty9b9t/Guadalajara_ZM_grid.gpkg?rlkey=w0b5jyk8296gaer2dbhug8yxf&st=1wpl6k4z&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/3o6sigukwuh7houty9b9t/Guadalajara_ZM_grid.gpkg?rlkey=w0b5jyk8296gaer2dbhug8yxf&st=7b9trc63&dl=1"
    },


    "Toluca": {
        "boundary": "https://www.dropbox.com/scl/fi/x7ousiufbx5nwl5x74bto/Toluca_ZM.gpkg?rlkey=elw2ncwapp4auhzqxwdmwrj4e&st=eto6woz5&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/mrrfds6q9sbybd1f00act/toluca_pedestrian_network.gpkg?rlkey=2rmv86oqdobfmsvm142vuc3p5&st=0l1y7cj8&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/b00774yduf2v41aphycck/ZM_toluca_pop_count.gpkg?rlkey=fhptvs6xk27152qaqrighgrx3&st=8yisa2hs&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/a99gcd0qirgm1hajpt6d9/ZM_toluca_poi_food.gpkg?rlkey=w1aljl0kt45k869kwub6d22bv&st=nydsc729&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/vrrxnoiaqahrorrhd2rde/Toluca_ZM_grid.gpkg?rlkey=1ju8mzlrolzr3mtbfnryj0vc5&st=4j2a8as0&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/yurrz9670bd4th51dk7vp/Toluca_ZM_centroids.gpkg?rlkey=igwpr2ftctva28bo4vio9s4xk&st=1dkytwlq&dl=1"
    },


    "Puebla-Tlaxcala": {
        "boundary": "https://www.dropbox.com/scl/fi/ugriorbjmkik5cbn2wv37/Puebla-Tlaxcala_ZM.gpkg?rlkey=p4hs2og0pkjkzch1phfhiytwr&st=x98uvwyr&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/j9sk44omxxivz1ny5809z/zacatecas_pedestrian_network.gpkg?rlkey=9n7mkh3lyfqs9f5jyak1ha69r&st=hvangc3i&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/5pi99ng04a80y73955l65/ZM_puebla-tlaxcala_pop_count.gpkg?rlkey=inlidxgrx006s33fl36722au8&st=wydtj805&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/kiondyp29elyzuax5lh3k/ZM_puebla-tlaxcala_poi_food.gpkg?rlkey=5235edy8kuuadoz58ej7cgn9b&st=oink616f&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/fm1slql9i3ykc9gqp4y65/Puebla-Tlaxcala_ZM_grid.gpkg?rlkey=m5b50dti53nfg9kmfuu48e7qx&st=sileju1l&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/t451nvx5eml3f1exve60k/Puebla-Tlaxcala_ZM_centroids.gpkg?rlkey=qx0a8ck5v3ynb4jr1o6vxh16q&st=5akxg2t2&dl=1"
    },


    "Merida": {
        "boundary": "https://www.dropbox.com/scl/fi/sfx5icr2o6hcvedc3s375/Merida_ZM.gpkg?rlkey=xft1cobuontr00ci2o7808cz8&st=ritq2j06&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/rxh87eyf32qodjcue9gij/merida_pedestrian_network.gpkg?rlkey=yjrjywxbew6b0nndsn1l0gx85&st=ymvu577h&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/ss2sb4ku8gq2n5le7v14t/ZM_merida_pop_count.gpkg?rlkey=nhs2fh2l39egvc9dr6xbyrsb4&st=xkwd14im&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/b551b063nmfai0rjnzs6g/ZM_merida_poi_food.gpkg?rlkey=a4glsyy0edi3v8am1lzzib5li&st=b1cqjt2x&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/6sarpx8rbnheh6waqqxpe/Merida_ZM_grid.gpkg?rlkey=306z45f8cmip99ny4zj5nb8pw&st=u7m42gek&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/mmcj20bd6nx1cwzbvpc8l/Merida_ZM_centroids.gpkg?rlkey=pgjx5k8ioz3jesmcds5vgs81e&st=d7o96o7f&dl=1"
    },


    "Tlaxcala-Apizaco": {
        "boundary": "https://www.dropbox.com/scl/fi/3zxzd839968tzrey5oe20/Tlaxcala-Apizaco_ZM.gpkg?rlkey=1lu0ok6w3q0b76rxyk6ugfiqf&st=pmp9vi7u&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/d01oru5la22m2gq5i5vua/Tlaxcala-Apizaco_pedestrian_network.gpkg?rlkey=dtgs800eagfgmdhnp9j1blfyl&st=y6yltf2i&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/jbovify8rc6pctdbueoen/ZM_Tlaxcala-Apizaco_pop_count.gpkg?rlkey=i14olk7rtfkx1egm0juxe2wx6&st=9m076a42&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/7gtgc6f22x6uz7ckdahaw/ZM_Tlaxcala-Apizaco_poi_food.gpkg?rlkey=9ltk3uepf3g8pnzi6a28y4him&st=xlc8gtez&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/q6x5t73xiv986gkupgytr/Tlaxcala-Apizaco_ZM_grid.gpkg?rlkey=ars2hy82q8fpz5ln0uy6k158n&st=v8jjeo47&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/fovc47lk6hh4ppajjl9ng/Tlaxcala-Apizaco_ZM_centroids.gpkg?rlkey=fccclewcpm4ei0z9jhs5f0gov&st=xxibtfzh&dl=1"
    },


    "Tijuana": {
        "boundary": "https://www.dropbox.com/scl/fi/uvip3ravz1ky2udyhhtk3/Tijuana_ZM.gpkg?rlkey=5ghpcfzjnh6h6mmsrcbv51327&st=8c97njjq&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/dc89efbp4iqohy989bto6/tijuana_pedestrian_network.gpkg?rlkey=eb7tpq2pr7opuuainzqby0huy&st=v4wg6iit&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/hziyx0gf2myncb4wilih3/ZM_tijuana_pop_count.gpkg?rlkey=6um1n6w6fbcekpu1k0vvu2nil&st=x9hy6cq9&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/dlih2n7kqknuccgd0s5ub/ZM_tijuana_poi_food.gpkg?rlkey=lunh13tjtnf6l6x8oetcfr3tb&st=usamjfrf&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/iemg8427nmml2hl2ji39i/Tijuana_ZM_grid.gpkg?rlkey=4jft3stjgedhe6yh8jg6oox9l&st=963o149j&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/wxb0vwfk4r43xlaqyr8jn/Tijuana_ZM_centroids.gpkg?rlkey=b0npl9dlabplfi7vtybjuxgsz&st=r8jz18gm&dl=1"
    },


    "Leon": {
        "boundary": "https://www.dropbox.com/scl/fi/23p3sebe9wwwycs3mo9ld/Leon_ZM.gpkg?rlkey=7v07bd6pqdhcztxjj6otlzwto&st=x3nqaima&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/cx3r1l4fzyxtuutba9epl/leon_pedestrian_network.gpkg?rlkey=jq4hiwqpjiiku5ebgn6j8tovq&st=9olygjkc&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/oyw4yhk4hx6bas3u3fxx7/ZM_leon_pop_count.gpkg?rlkey=di2u4wmvz6mlvdav69xnsbo2a&st=jsqx02kk&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/sqdvgujveaoewyoku9sd0/ZM_leon_poi_food.gpkg?rlkey=wqhiql4y2z1o77t7wsmfzmmoz&st=7mbknyzv&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/vyfk9spgc7q9mi343uoh4/Leon_ZM_grid.gpkg?rlkey=t5qbj5ovnzizweok7wp9k5lya&st=k082s705&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/a26cf9vyqq3riz4nurrxx/Leon_ZM_centroids.gpkg?rlkey=x9qrioj9eqsjkctx40c3i81ea&st=w79si7ia&dl=1"
    },


    "La Laguna": {
        "boundary": "https://www.dropbox.com/scl/fi/rtj11ma1g7ev0prpl0elf/Laguna_ZM.gpkg?rlkey=s2953b77p39sp04p2kl7wu936&st=foky2ig5&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/gcg6eu9i5pyyyowqu8m2g/laguna_pedestrian_network.gpkg?rlkey=qid3gws0ic2rdfyj3lz2eh7tl&st=09gtuge1&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/obauyohh8svjhu7z9olb3/ZM_laguna_pop_count.gpkg?rlkey=ndb7rcpd25mmuzurl0r5ws41g&st=847603tg&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/ku96inlr4kvrtyqot9x98/ZM_laguna_poi_food.gpkg?rlkey=blfgcrfwq5selsg2l18w7ft1i&st=q7jhyink&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/jr18ol0vv5k5ekymvivgd/Laguna_ZM_grid.gpkg?rlkey=vnqlmympbnu68sfuvtdinoxv4&st=qclq6g1c&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/u7u7z7x4fol9i0ndacqx4/Laguna_ZM_centroids.gpkg?rlkey=lyux2495vn5amxt1v7jlr6ozt&st=1zwyydi2&dl=1"
    },


    "Queretaro": {
        "boundary": "https://www.dropbox.com/scl/fi/h5oo7jrfkg146i9urm0i7/Queretaro_ZM.gpkg?rlkey=1hs3pn4me5mnqveg8xdlhfutx&st=qhccuqv2&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/fal6u5a1jb7igr0x012e2/queretaro_pedestrian_network.gpkg?rlkey=3zgofsbc0oj2a03l9nqc2iq96&st=h1bx74hs&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/t0qa7zijv60dad1rodqag/ZM_queretaro_pop_count.gpkg?rlkey=eheg5lt9j2aguuf4ofvzhj4l3&st=fsrvgpb7&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/rbfyiq7f00dvpccca7r4m/ZM_queretaro_poi_food.gpkg?rlkey=4a1ne13mruvgb64p7onoele0j&st=yhqlku75&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/fkhlsb226d1x9spirc47l/Queretaro_ZM_grid.gpkg?rlkey=eemz0bxofob6r62n23x411kvc&st=4we64nw7&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/w48l1qtoe3yyfzhsf7lq6/Queretaro_ZM_centroids.gpkg?rlkey=hx93d23vigyfnb0p4xy83y416&st=6mwlmm27&dl=1"
    },


    "Chihuahua": {
        "boundary": "https://www.dropbox.com/scl/fi/12tstdj7hwde7z79j2tqd/Chihuahua_ZM.gpkg?rlkey=iu7pqm2ezz1w9oqcf8zkrewe8&st=fq9nk545&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/fnsb4orldwj1z3vdbxnil/chihuahua_pedestrian_network.gpkg?rlkey=xhoilw6jkr3g9h9budgp3h5ki&st=ofv70l10&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/ir3l3wmc9lxfh9c75lfi1/ZM_chihuahua_pop_count.gpkg?rlkey=h7pjobhgb69s2oe9ihfsxohkj&st=kyan05c2&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/enb3q6jv9bi8f5yp5htyd/ZM_chihuahua_poi_food.gpkg?rlkey=ke4dkbsq566kts5k6hcs1845q&st=xa13yjf0&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/yzl70caf6j8tlc7cfvhrm/Chihuahua_ZM_grid.gpkg?rlkey=bpwgk3n391obudlygf7vbvncm&st=aq0ye21k&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/w8jofj0fvat26kv1r126s/Chihuahua_ZM_centroids.gpkg?rlkey=2lnr36p531vaiso6u5l0a24hf&st=9djqmmay&dl=1"
    },


    "Saltillo": {
        "boundary": "https://www.dropbox.com/scl/fi/50llro911fj53q1mqmgzh/Saltillo_ZM.gpkg?rlkey=evmgp9c3wot0d93ood7c3gi1w&st=2gj802h9&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/siqif4adlwlt8oe8zv1st/saltillo_pedestrian_network.gpkg?rlkey=b9rnc2fb2brtuf4i4qgntf316&st=6s1qw2dj&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/q7jbor8xdws7wr43o31d9/ZM_saltillo_pop_count.gpkg?rlkey=w42jpv1ui7cek0dnix9m9t2o9&st=jemkr6xs&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/cbthzx3ugw06q7a88wwb4/ZM_saltillo_poi_food.gpkg?rlkey=znkpipezvuet7vfl38795rdkq&st=lq9n7ofy&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/m5mjmsrzsh0h85fwu48h2/Saltillo_ZM_grid.gpkg?rlkey=tzkquyj02lz53qqbq6xybmwj8&st=xtht1qys&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/sp0p3dchkwewrlbh1nkk7/Saltillo_ZM_centroids.gpkg?rlkey=wqu0gpr8iz7w1vaa35yiygxgc&st=9kwk19gj&dl=1"
    },


    "Aguascalientes": {
        "boundary": "https://www.dropbox.com/scl/fi/kw674cefpxo8wij0zdqe9/Aguascalientes_ZM.gpkg?rlkey=j45a2sd79o55w7pnlgphbwy59&st=jd6dcv1o&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/9tq849alcusg2ag47sraa/aguascalientes_pedestrian_network.gpkg?rlkey=xsh2jawtod9ggekfhtmkpftt0&st=wcadiz91&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/hucfhjo92ei2vi0ja9zuo/ZM_aguascalientes_pop_count.gpkg?rlkey=8jgsfya7uqbevm3tfrue95dhe&st=dkvh1wx1&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/yp439bd52sdzhweny7w9n/ZM_aguascalientes_poi_food.gpkg?rlkey=ga9x5skkdq48qqnxvmgklrm35&st=31yeltcr&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/xuzfryq9mgncwbx43xpaw/Aguascalientes_ZM_grid.gpkg?rlkey=rwyzgaaezu7bicidnb0tra2hj&st=p3m4y2di&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/qdzmnup9hco7k3p4bkji3/Aguascalientes_ZM_centroids.gpkg?rlkey=aqidp949rus1k8yvawraaesui&st=2j8bxdc1&dl=1"
    }
}

    # Process each city
    results = {}
    total_cities = len(urls)
    
    for idx, (city_name, city_urls) in enumerate(urls.items(), 1):
        print(f"\n\n{'#'*80}")
        print(f"# CITY {idx}/{total_cities}: {city_name}")
        print(f"{'#'*80}")
        
        try:
            result_grid = process_city(city_name, city_urls)
            results[city_name] = result_grid
        except Exception as e:
            print(f"\n❌ ERROR processing {city_name}: {e}")
            print("Continuing with next city...")
            continue
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - ALL CITIES")
    print(f"{'='*80}")
    print(f"Total cities processed: {len(results)}/{total_cities}")
    for city_name in results.keys():
        print(f"  ✅ {city_name}")
    
    if len(results) < total_cities:
        failed_cities = set(urls.keys()) - set(results.keys())
        print(f"\nFailed cities: {len(failed_cities)}")
        for city_name in failed_cities:
            print(f"  ❌ {city_name}")
    
    print(f"\n{'='*80}")
    print("✅ Multi-city analysis completed!")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    results = main()
    result_grid = main()