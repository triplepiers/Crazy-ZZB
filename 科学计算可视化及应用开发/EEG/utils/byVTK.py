import vtk
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolygon,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTextActor
)

from vtk import vtkScalarBarActor
from .MSquare import marching_squares

def createClrTable(range):
    clrTable = vtk.vtkLookupTable()
    clrTable.SetNumberOfColors(512)
    clrTable.SetHueRange(1.0, 0.0)
    clrTable.SetTableRange(*range)
    clrTable.Build()
    return clrTable


def createPtSet(pSt, N):
    for i in range(N+1):
        for j in range(N+1):
            pSt.InsertNextPoint(float(i), float(j), 0.0)
    return


def createPolygon(i, j, N):
    polygon = vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    lb = i*(N+1)+j                           # Left_Btm idx
    # LB -> RB -> RT -> LT
    polygon.GetPointIds().SetId(0, lb)
    polygon.GetPointIds().SetId(1, lb+1)
    polygon.GetPointIds().SetId(2, lb+N+2)
    polygon.GetPointIds().SetId(3, lb+N+1)
    return polygon

def createBarActor(clrTable):
    barActor = vtkScalarBarActor()
    barActor.SetNumberOfLabels(5)
    barActor.SetLookupTable(clrTable)
    barActor.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)
    # set Pos
    barActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    barActor.GetPositionCoordinate().SetValue(0.05, 0.1)
    barActor.SetWidth(0.1)
    barActor.SetHeight(0.8)
    return barActor


def createChannelActor(X, Y, point_size=5):
    chPSt = vtkPoints()
    vertices = vtkCellArray()
    for x, y in zip(X, Y):
            pid = chPSt.InsertNextPoint(x, y, 0.0)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)

    polydata = vtkPolyData()
    polydata.SetPoints(chPSt)
    polydata.SetVerts(vertices)
    polydata.Modified()

    ptMapper = vtkPolyDataMapper()
    ptMapper.SetInputData(polydata)

    actor = vtkActor()
    actor.SetMapper(ptMapper)
    actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('Black'))
    actor.GetProperty().SetPointSize(point_size)
    return actor


def createTextActor(tar):
    txtActor = vtkTextActor()
    txtActor.SetInput(f'contour = {tar:.2f}')
    txtActor.SetPosition2(40, 40)
    txtActor.GetProperty().SetColor(vtkNamedColors().GetColor3d('Black'))
    txtActor.GetTextProperty().SetFontSize(30)
    return txtActor

def createSphere(r):
    sphere = vtk.vtkSphere()
    sphere.SetCenter(r, r, 0)
    sphere.SetRadius(r+0.5)
    return sphere


# 使用 Sphere 裁切平面，保留圆形数据
def clip(plane, clip):
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(plane)
    clipper.SetClipFunction(clip)
    clipper.GenerateClippedOutputOn() # 保留交集
    clipper.Update()
    return clipper.GetClippedOutput()


def vis(W, N, title, X, Y, contour: int=0.6):

    # Setup four points
    points = vtkPoints()
    createPtSet(points, N)

    # Create the polygons & Add the polys to a list
    polygons = vtkCellArray()
    for i in range(N):
        for j in range(N):
            p = createPolygon(i, j, N)
            polygons.InsertNextCell(p)

    # Set Scalars (as W)
    scalars = vtk.vtkFloatArray()
    flt_W = W.flatten()
    for i in range((N + 1) ** 2):
        scalars.InsertTuple1(i, flt_W[i])

    # Create a PolyData
    planePolyData = vtkPolyData()
    planePolyData.SetPoints(points)
    planePolyData.SetPolys(polygons)
    planePolyData.GetPointData().SetScalars(scalars)

    # clip
    polygonPolyData = clip(planePolyData, createSphere(N/2))

    # Create a mapper and actor
    mapper = vtkPolyDataMapper()
    mapper.SetScalarRange(scalars.GetRange())
    # 自定义颜色
    clrTable = createClrTable(scalars.GetRange())
    mapper.SetLookupTable(clrTable)
    mapper.SetInputData(polygonPolyData)

    actor = vtkActor()
    actor.SetMapper(mapper)

    barActor = createBarActor(clrTable)
    chActor = createChannelActor(X, Y)
    lineActor = marching_squares(W, tar=contour)
    txtActor = createTextActor(tar=contour)

    # Draw within a window
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetSize(550, 550)
    renderWindow.AddRenderer(renderer)

    renderer.AddActor(actor)
    renderer.AddActor(chActor)
    renderer.AddActor(lineActor)
    renderer.AddActor(txtActor)
    renderer.AddActor2D(barActor)
    renderer.SetBackground(vtkNamedColors().GetColor3d('White'))
    renderWindow.Render()

    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetInteractorStyle(None)  # 暂时关掉鼠标交互
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Start()




