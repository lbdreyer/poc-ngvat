import netCDF4 as nc
import numpy as np
import pyvista as pv
import vtk


def mesh_from_nc(fname='qrclim.sst.ugrid.nc', data_name='surface_temperature'):
    """Load pyvista mesh from netcdf file."""
    ds = nc.Dataset(fname)

    # Get data and face_node_connectivity variables
    data_var = ds.variables['surface_temperature']
    face_node_var = ds.variables['dynamics_face_nodes']

    node_face = face_node_var[:].data
    data = data_var[:].data[0] # SELECT 1ST TIME STEP!
    
    # Get the Fill Value
    fill_value = data_var._FillValue
    # Get start index
    start_index = face_node_var.start_index

    # Get the lat, lons
    node_x = ds.variables['dynamics_node_x'][:].data
    node_y = ds.variables['dynamics_node_y'][:].data

    # Convert longitudes from (0, 360) to (-180, 180)
    node_x[node_x > 180] -=360

    # Face_node connectivity may be 1 based counting if start index is 1
    node_face -= start_index
    
    # Skip faces that straddle the -180/+180 line. (Causes problems with plotting!)
    diff = node_x[node_face].max(axis=1) - node_x[node_face].min(axis=1)
    diff_less_90 = diff < 90
    node_face = node_face[diff_less_90]
    data = data[diff_less_90]

    # Skip faces with data that matches missing data value
    real_data = data != fill_value
    node_face = node_face[real_data]
    data = data[real_data]
    
    # Create PolyData
    z_zeros = np.zeros(len(node_x))
    vertices = np.vstack([node_x, node_y, z_zeros]).T
    face_4 = 4*np.ones(len(node_face)).reshape((len(node_face), 1))
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
