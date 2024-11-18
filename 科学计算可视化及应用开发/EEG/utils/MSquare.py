from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkLine
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)

contourActor = vtkActor()
polyData = vtkPolyData()
mapper = vtkPolyDataMapper()
mapper.SetInputData(polyData)
contourActor.SetMapper(mapper)
contourActor.GetProperty().SetLineWidth(4)
contourActor.GetProperty().SetColor(vtkNamedColors().GetColor3d('Black'))

def draw_all(pts, lines):
    polyData.SetPoints(pts)
    polyData.SetLines(lines)
    contourActor.Modified()
    return


def marching_squares(grid_data, tar):
    def linear_interpolation(a, b):
        return abs((tar - a) / (a - b))

    pts = vtkPoints()
    lines = vtkCellArray()

    def create_line(x1, y1, x2, y2):
        line = vtkLine()
        p1, p2 = [x1, y1, 0.0], [x2, y2, 0.0]
        line.GetPointIds().SetId(0, pts.InsertNextPoint(p1))
        line.GetPointIds().SetId(1, pts.InsertNextPoint(p2))
        lines.InsertNextCell(line)
        return

    height, width = grid_data.shape
    for i in range(width-1):
        for j in range(height-1):
            cell = 0
            # LB -> RB -> RT -> LT
            c0 = grid_data[i][j]
            if c0 > tar: cell += 8
            c1 = grid_data[i+1][j]
            if c1 > tar: cell += 4
            c2 = grid_data[i+1][j+1]
            if c2 > tar: cell += 2
            c3 = grid_data[i][j+1]
            if c3 > tar: cell += 1

            if cell == 0 or cell == 15: pass
            if cell == 1:
                off1 = linear_interpolation(c3, c2)
                off2 = linear_interpolation(c0, c3)
                create_line(i + off1, j + 1, i, j + off2)
            elif cell == 2:
                off1 = linear_interpolation(c3, c2)
                off2 = linear_interpolation(c1, c2)
                create_line(i + off1, j + 1, i + 1, j + off2)
            elif cell == 3:
                off1 = linear_interpolation(c0, c3)
                off2 = linear_interpolation(c1, c2)
                create_line(i, j + off1, i + 1, j + off2)
            elif cell == 4:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c1, c2)
                create_line(i + off1, j, i + 1, j + off2)
            elif cell == 5:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c1, c2)
                off3 = linear_interpolation(c3, c2)
                off4 = linear_interpolation(c0, c3)
                create_line(i, j + off4, i + off1, j)
                create_line(i + off3, j + 1, i + 1, j + off2)
            elif cell == 6:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c3, c2)
                create_line(i + off1, j, i + off2, j + 1)
            elif cell == 7:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c0, c3)
                create_line(i, j + off2, i + off1, j)
            elif cell == 8:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c0, c3)
                create_line(i, j + off2, i + off1, j)
            elif cell == 9:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c3, c2)
                create_line(i + off1, j, i + off2, j + 1)
            elif cell == 10:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c1, c2)
                off3 = linear_interpolation(c3, c2)
                off4 = linear_interpolation(c0, c3)
                create_line(i + off1, j, i + 1, j + off2)
                create_line(i, j + off4, i + off3, j + 1)
            elif cell == 11:
                off1 = linear_interpolation(c0, c1)
                off2 = linear_interpolation(c1, c2)
                create_line(i + off1, j, i + 1, j + off2)
            elif cell == 12:
                off1 = linear_interpolation(c0, c3)
                off2 = linear_interpolation(c1, c2)
                create_line(i, j + off1, i + 1, j + off2)
            elif cell == 13:
                off1 = linear_interpolation(c1, c2)
                off2 = linear_interpolation(c3, c2)
                create_line(i + off2, j + 1, i + 1, j + off1)
            elif cell == 14:
                off1 = linear_interpolation(c0, c3)
                off2 = linear_interpolation(c3, c2)
                create_line(i, j + off1, i + off2, j + 1)

    draw_all(pts, lines)
    return