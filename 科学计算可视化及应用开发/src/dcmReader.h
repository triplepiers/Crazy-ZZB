#define DATA_BASE_DIR "/../data"

#include <vtkDICOMImageReader.h>
#include <vtkSmartPointer.h>
#include <vtkImageViewer2.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageData.h>
#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <direct.h>
#include <io.h>
#else
#include <unistd.h>
#include <dirent.h>
#endif

#include <iostream>
#include <string>
#include <vector>

using namespace std;

void readDCM(string inputFile, vtkSmartPointer<vtkDICOMImageReader> &reader);
void showDCM(vtkSmartPointer<vtkImageData> &imageData, vtkSmartPointer<vtkDICOMImageReader> &reader);
string getCWD(void);
vector<string> getDCMlist(string data_dir);