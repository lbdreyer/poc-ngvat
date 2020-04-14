import os

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


def project_mesh(mesh, proj_name='moll'):
    """
    Returns the input mesh, reprojecte into the target projection, proj_name
    
    Note currently this uses two workarounds that should be fixed at some point
    1) I was unsure how to get hold of the low-level vtkPolyData object from the
       pyvista.PolyData object, so currently I save it out to a .vtk file, then
       load it back in using vtk directly.
    2) 
    
    """
    fname = 'temp.vtk'
    
    mesh.save(fname)
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.ReadAllScalarsOn()
    output = reader.GetOutputPort()

    # Set up source and target projection
    sourceProjection = vtk.vtkGeoProjection()
    destinationProjection = vtk.vtkGeoProjection()
    destinationProjection.SetName(proj_name)
    
    # Set up transform between source and target.
    transformProjection = vtk.vtkGeoTransform()
    transformProjection.SetSourceProjection(sourceProjection)
    transformProjection.SetDestinationProjection(destinationProjection)

    # Set up transform filter
    transformGraticle = vtk.vtkTransformPolyDataFilter()
    transformGraticle.SetInputConnection(output)
    transformGraticle.SetTransform(transformProjection)

    # Plot it vtk first to get it working. TODO: skip this?
    result = transformGraticle.GetOutputPort()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(result)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    iren = vtk.vtkRenderWindowInteractor()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetInteractor(iren)
    iren.Initialize()
    renWin.Render()
    del renWin
    del iren

    # Wrap output of transform as a Pyvista object.
    proj_mesh = pv.wrap(transformGraticle.GetOutput())
    
    return proj_mesh
