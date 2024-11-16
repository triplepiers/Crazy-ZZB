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
    vtkPolygon
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

from vtk import vtkScalarBarActor


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



def vis(W, N, title: str = ""):
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

    # extract = vtk.vtkMarchingSquares()
    # extract.SetInputArrayToProcess()

    # Create a PolyData
    polygonPolyData = vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    polygonPolyData.GetPointData().SetScalars(scalars)

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

    # Draw within a window
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.SetSize(550, 550)
    renderWindow.AddRenderer(renderer)

    renderer.AddActor(actor)
    renderer.AddActor2D(barActor)
    renderer.SetBackground(vtkNamedColors().GetColor3d('White'))
    renderWindow.Render()

    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetInteractorStyle(None)  # 暂时关掉鼠标交互
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Start()




