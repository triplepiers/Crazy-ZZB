#include "dcmReader.h"

// read Single DCM Image
void readDCM(string inputFile, vtkSmartPointer<vtkDICOMImageReader> &reader)
{
  reader = vtkSmartPointer<vtkDICOMImageReader>::New();

  reader->SetFileName(inputFile.c_str());
  reader->Update();

  vtkSmartPointer<vtkImageData> imageData = reader->GetOutput();
  showDCM(imageData, reader);
  return;
}

// show Single DCM Image
void showDCM(vtkSmartPointer<vtkImageData> &imageData, vtkSmartPointer<vtkDICOMImageReader> &reader) {
  vtkSmartPointer<vtkImageViewer2> imageViewer = vtkSmartPointer<vtkImageViewer2>::New();
  imageViewer->SetInputConnection(reader->GetOutputPort());

  vtkSmartPointer<vtkRenderWindowInteractor> renWin;
  renWin = vtkSmartPointer<vtkRenderWindowInteractor>::New();

  imageViewer->SetupInteractor(renWin);
  imageViewer->Render();
  renWin->Start();
  return;
}

string getCWD(void)
{
  char *buffer = NULL;
#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
  buffer = _getcwd(NULL, 0);
#else
  buffer = getcwd(NULL, 0);
#endif

  if (buffer)
  {
    string cwd = buffer;
    free(buffer);
    return cwd;
  }
  else
  {
    return "";
  }
}

vector<string> getDCMlist(string data_dir) {
  vector<string> DCMlist = {};
#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
  // todo
#else
  DIR *dp = opendir(data_dir.c_str());
  if (!dp) {
    cout << "Error opening directory " << data_dir << endl;
    exit(1);
  } else {
    struct dirent *dirp;
    while ((dirp = readdir(dp)) != NULL) {
      string file_name = dirp->d_name;
      if (file_name == "." || file_name == "..") continue;
      else                                       DCMlist.push_back(file_name);
    }
  }
#endif
  return DCMlist;
}

int main(void)
{
  string data_dir = getCWD() + DATA_BASE_DIR;
  cout << "reading data from dir: " << data_dir << endl;

  vector<string> file_list = getDCMlist(data_dir);
  vtkSmartPointer<vtkDICOMImageReader> reader = vtkSmartPointer<vtkDICOMImageReader>::New();
  for (vector<string>::iterator it = file_list.begin(); it != file_list.end(); it++) {
    string file_url = data_dir + "/" + *it;
    cout << "file_url: " << file_url << endl;
    readDCM(file_url, reader);
    break;
  }
  return 0;
}