import os

import cartopy
import cartopy.io.shapereader as shp
import netCDF4 as nc
import numpy as np
import pyvista as pv
import vtk


def mesh_from_nc(fname='qrclim.sst.ugrid.nc', data_type='real', which='new'):
    """
    Load pyvista mesh from netcdf file.
    
    which - old = skips the data straddling the line, new - adds extra data.
    
    """
    ds = nc.Dataset(fname)

    if data_type != 'real':
        c_res = os.path.basename(fname).split('_')[1][1:]
        if c_res == '4':
            data_name, x, y, face = ('synthetic', f'example_C{c_res}_node_x',
                                f'example_C{c_res}_node_y', f'example_C{c_res}_face_nodes')
        else:
            data_name, x, y, face = ('synthetic', 'dynamics_node_x',
                                'dynamics_node_y', 'dynamics_face_nodes')
    else:
        data_name, x, y, face = ('surface_temperature', 'dynamics_node_x',
                            'dynamics_node_y', 'dynamics_face_nodes')

    
    # Get data and face_node_connectivity variables
    data_var = ds.variables[data_name]
    face_node_var = ds.variables[face]

    node_face = face_node_var[:].data

    data = data_var[:].data
    if data_type == 'real':
        data = data[0] # SELECT 1ST TIME STEP!
        # Get the Fill Value
        fill_value = data_var._FillValue
        # Skip faces with data that matches missing data value
        real_data = data != fill_value
        node_face = node_face[real_data]
        data = data[real_data]
       
    # Get start index
    start_index = face_node_var.start_index

    # Get the lat, lons
    node_x = ds.variables[x][:].data
    node_y = ds.variables[y][:].data

    # Convert longitudes from (0, 360) to (-180, 180)
    node_x[node_x > 180] -=360

    # Face_node connectivity may be 1 based counting if start index is 1
    node_face -= start_index
    
    if which=='old':
        # Skip faces that straddle the -180/+180 line. (Causes problems with plotting!)
        diff = node_x[node_face].max(axis=1) - node_x[node_face].min(axis=1)
        diff_less_90 = diff < 90
        node_face = node_face[diff_less_90]
        data = data[diff_less_90]
    
    else:
        # Skip faces that straddle the -180/+180 line. (Causes problems with plotting!)
        diff = node_x[node_face].max(axis=1) - node_x[node_face].min(axis=1)
        locs = node_x[node_face]
        diff_more_90 = diff >180
        bad_x = locs[diff_more_90]
        bad_x[bad_x == 180] = -180
        new_cells_num = len(bad_x)
        extra_x = np.ravel(bad_x)
        extra_y = np.ravel(node_y[node_face[diff_more_90]])
        extra_node_face = np.arange(new_cells_num * 4).reshape((len(bad_x), 4)) + len(node_x)
        extra_data = data[diff_more_90]

        diff_less_90 = diff <= 180
        node_face = node_face[diff_less_90]
        data = data[diff_less_90]

        # Append extra *corrected cells*
        node_x = np.concatenate([node_x, extra_x])
        node_y = np.concatenate([node_y, extra_y])
        node_face = np.concatenate([node_face, extra_node_face])
        data = np.concatenate([data, extra_data])
   
    # Create PolyData
    z_zeros = np.zeros(len(node_x))
    vertices = np.vstack([node_x, node_y, z_zeros]).T
    face_4 = 4 * np.ones(len(node_face)).reshape((len(node_face), 1))
    faces = np.hstack([face_4[:], node_face[:]])
    faces = np.ravel(faces)
    mesh = pv.PolyData(vertices, faces)
   
    # Add data to cell faces
    mesh.cell_arrays[data_name] = data[:]
    
    return mesh


class PolydataTransformFilter:
    def __init__(self, proj_name='moll'):
        # Set up source and target projection
        sourceProjection = vtk.vtkGeoProjection()
        destinationProjection = vtk.vtkGeoProjection()
        destinationProjection.SetName(proj_name)

        # Set up transform between source and target.
        transformProjection = vtk.vtkGeoTransform()
        transformProjection.SetSourceProjection(sourceProjection)
        transformProjection.SetDestinationProjection(destinationProjection)

        # Set up transform filter
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transformProjection)
        
        self.transform_filter = transform_filter
    
    def transform(self, mesh):
        self.transform_filter.SetInputData(mesh)
        self.transform_filter.Update()
        output = self.transform_filter.GetOutput()
        
        # Wrap output of transform as a Pyvista object.
        return pv.wrap(output)


def get_coastlines(resolution="110m"):
    """
    Modified version of 
    https://github.com/bjlittle/poc-ngvat/blob/master/poc-3/utils.py
    """
    fname = shp.natural_earth(resolution=resolution, category="physical", name="coastline")
    reader = shp.Reader(fname)

    dtype = np.float32
    blocks = pv.MultiBlock()

    for i, record in enumerate(reader.records()):
        for geometry in record.geometry:
            xy = np.array(geometry.coords[:], dtype=dtype)
            x = xy[:,0].reshape(-1, 1)
            y = xy[:,1].reshape(-1, 1)
            z = np.zeros_like(x)

            xyz = np.hstack((x, y, z))
            poly = pv.lines_from_points(xyz, close=False)
            blocks.append(poly)

    return blocks
