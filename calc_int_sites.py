import numpy as np
from pymatgen.analysis.defects.generators import  VoronoiInterstitialGenerator, ChargeInterstitialGenerator
from pymatgen.io.vasp import Poscar, Chgcar
from pymatgen.core import Element
from pymatgen.io.cif import CifWriter

# USER INPUTS
#-------------------------------------------
interstitial_element_to_place = "N"
number_of_interstitials_to_insert = 5
which_interstitial_to_use = 0 # The value '0' will consider all found available interstitial positions for calculating
#the distances. If you want to place interstitials on only the certain type of interstitial site, e.g., this method will
#identify two different types of interstitial positions and you want to insert interstitial only on the first type,
#then change the value for this variable to the number 1. If you want to place only on the second interstitial site type,
#change it to value 2.
#-------------------------------------------



structure = Poscar.from_file("POSCAR").structure
generator = VoronoiInterstitialGenerator(
    clustering_tol=0.75,
    min_dist=0.5,
)

"""
If you want to find the interstitial sites based on the charge density method instead of Voronoi method,
uncomment the following two variables.This approach requires the charge density CHGCAR file from VASP as input
(instead of the structural POSCAR file).
Ensure you generate the CHGCAR file first by performing a VASP calculation on your initial structure

generator = ChargeInterstitialGenerator( clustering_tol=0.75,
    min_dist=0.5)
structure =  Chgcar.from_file("CHGCAR")
"""

"""
FIRST PART: Obtain all interstitial sites with fractional coordinates.
"""
frac_coords = []
unique_int = []
unique_mult = []
idx = 0
frac_coords_dict = {}
for interstitial in generator.generate(structure, "H"): #Element 'H' is here only to find the available sites in order to prevent error with oxidation states for some elements like noble gases
    frac_coords_dict[idx]=[]
    print(f"\nUnique interstitial site at: {interstitial.site.frac_coords}")
    print(f"It has multiplicity of: {interstitial.multiplicity}")
    print(f"--------------------------------------------------\n\n")
    print(f"The following are all the equivalent positions (lattice coordinates) [fractional coordinates]\n:")
    unique_int.append(interstitial.site.frac_coords)
    unique_mult.append(interstitial.multiplicity)
    for site in interstitial.equivalent_sites:
     print(f"Fractional: {site.frac_coords}")
     frac_coords.append(site.frac_coords)
     frac_coords_dict[idx].append(site.frac_coords)
    idx = idx + 1
print(f"\nThere are total of {len(unique_int)} unique interstitial sites at\n:{unique_int}, having multiplicity of {unique_mult})\n.If you wish to place interstitials only at certain type of interstitial positions, change the 'which_interstitial_to_use")

if which_interstitial_to_use == 0:
        frac_coords_use = frac_coords
else:
        frac_coords_use = frac_coords_dict[which_interstitial_to_use-1]

      
def wrap_coordinates(frac_coords):
    """
    Wrap fractional coordinates into the range [0, 1).
    """
    frac_coords = np.array(frac_coords)  # Ensure input is a NumPy array
    return frac_coords % 1


def compute_periodic_distance_matrix(frac_coords):
    """
    Compute a periodic distance matrix for a set of fractional coordinates.

    Args:
        frac_coords (ndarray): Fractional coordinates of shape (N, 3).

    Returns:
        ndarray: Distance matrix of shape (N, N).
    """
    n = len(frac_coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            delta = frac_coords[i] - frac_coords[j]
            delta = delta - np.round(delta)  # Apply periodic boundary conditions
            dist_matrix[i, j] = dist_matrix[j, i] = np.linalg.norm(delta)
    return dist_matrix


def select_spaced_points(frac_coords, n_points=5, mode="farthest", target_value=0.5):
    """
    Select n_points that are maximally spaced apart under periodic boundary conditions.

    Args:
        frac_coords (list of list of float): Fractional coordinates of points.
        n_points (int): Number of points to select.

    Returns:
        list of list of float: Selected fractional coordinates.
    """
    frac_coords = wrap_coordinates(frac_coords)  # Wrap fractional coordinates
    dist_matrix = compute_periodic_distance_matrix(frac_coords)
    # Greedy selection of points
    selected_indices = [0]  # Start with the first point
    for _ in range(1, n_points):
        remaining_indices = [i for i in range(len(frac_coords)) if i not in selected_indices]

        if mode == "farthest":
            # Select the point that is farthest from already selected points
            next_index = max(
                remaining_indices,
                key=lambda i: min(dist_matrix[i, j] for j in selected_indices)
            )
        elif mode == "nearest":
            # Select the point that is nearest to already selected points
            next_index = min(
                remaining_indices,
                key=lambda i: min(dist_matrix[i, j] for j in selected_indices)
            )
        elif mode == "moderate":
            # Select the point with average distance closest to the target_value
            next_index = min(
                remaining_indices,
                key=lambda i: abs(sum(dist_matrix[i, j] for j in selected_indices) / len(selected_indices) - target_value)
            )
        else:
            raise ValueError("Invalid mode. Choose from 'nearest', 'farthest', or 'moderate'.")

        selected_indices.append(next_index)

    return frac_coords[selected_indices].tolist()



selected_points_farthest = select_spaced_points(frac_coords_use, n_points=number_of_interstitials_to_insert, mode = 'farthest')
selected_points_nearest = select_spaced_points(frac_coords_use, n_points=number_of_interstitials_to_insert, mode = 'nearest')
selected_points_moderate = select_spaced_points(frac_coords_use, n_points=number_of_interstitials_to_insert, mode = 'moderate')



structure_to_save = Poscar.from_file("POSCAR").structure
print("Saving structure with FARTHEST distances between interstitials into POSCAR and CIF files...")
for point in selected_points_farthest:
    structure_to_save.append(
        species=Element(interstitial_element_to_place),
        coords=point,
        coords_are_cartesian=False  # Specify that the coordinates are fractional
     )

cif_writer = CifWriter(structure_to_save)
cif_writer.write_file("modified_structure_farthest.cif")  # Output CIF file
poscar = Poscar(structure_to_save)
poscar.write_file("modified_structure_farthest.POSCAR")  # Output POSCAR file


structure_to_save = Poscar.from_file("POSCAR").structure
print("Saving structure with NEAREST distances between interstitials into POSCAR and CIF files...")
for point in selected_points_nearest:
    structure_to_save.append(
        species=Element(interstitial_element_to_place),
        coords=point,
        coords_are_cartesian=False  # Specify that the coordinates are fractional
     )

cif_writer = CifWriter(structure_to_save)
cif_writer.write_file("modified_structure_nearest.cif")  # Output CIF file
poscar = Poscar(structure_to_save)
poscar.write_file("modified_structure_nearest.POSCAR")  # Output POSCAR file


structure_to_save = Poscar.from_file("POSCAR").structure
print("Saving structure with MODERATE distances between interstitials into POSCAR and CIF files...")
for point in selected_points_moderate:
    structure_to_save.append(
        species=Element(interstitial_element_to_place),
        coords=point,
        coords_are_cartesian=False  # Specify that the coordinates are fractional
     )

cif_writer = CifWriter(structure_to_save)
cif_writer.write_file("modified_structure_moderate.cif")  # Output CIF file
poscar = Poscar(structure_to_save)
poscar.write_file("modified_structure_moderate.POSCAR")  # Output POSCAR file
