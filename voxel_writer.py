import numpy as np
import base64


VTK_TO_ABAQUS_TYPE = {
    8: "S4R",
    9: "S4R",
    10: "C3D4",
    11: "C3D8R",
    12: "C3D8R"
}

class Voxel_writer():
    
    '''
    def __init__(self, filename, type, nodes, elements):
        self._filename = filename 
        self._type = type
        self._nodes = nodes
        self._elements = elements
    '''
    def __init__(self, params, nodes, elements):
        self._filename = params.filepath
        self._type = params.type
        self._nodes = nodes
        self._elements = elements

    def write_inp(self):
        ele_offset = 1
        with open(self._filename, "wt") as f:
            f.write("*HEADING\n")
            f.write('\n*Node\n')
            fmt = ", ".join(["{}"] + ["{:.9f}"] * self._nodes.shape[1]) + "\n"
            for k, x in enumerate(self._nodes):
                f.write(fmt.format(k + 1, *x))
            
            ele_nod = self._elements + ele_offset
            f.write('\n*Element, type={}\n'.format(VTK_TO_ABAQUS_TYPE[self._type]))

            for e in range(1, self._elements.shape[0] + 1):
                f.write('{:d},  {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(e,*ele_nod[e-1,:]))
        return

    def write_vtk(self):
        with open(self._filename, "wt") as f:
            f.write(
                "# vtk DataFile Version 5.1\n"
                "Volume Mesh\n"
                "ASCII\n"
                "DATASET UNSTRUCTURED_GRID\n")

            f.write("POINTS  {:d}  float\n".format(self.__getattribute__nodes.shape[0]) )
            self._nodes.tofile(f, sep=" ")
            f.write("\n")

            f.write("CELLS {:d} {:d}\n".format(self._elements.shape[0]+1, self._elements.size))
            f.write("OFFSETS vtktypeint64\n")
            offsets = np.arange(0, self._elements.size+1, self._elements.shape[1], dtype=int)
            offsets.tofile(f, sep="\n")
            f.write("\n")

            f.write("CONNECTIVITY vtktypeint64\n")
            self._elements.tofile(f, sep="\n")
            f.write("\n")

            f.write("CELL_TYPES  {:d}\n".format(self._elements.shape[0]))
            np.full(self._elements.shape[0], self._type).tofile(f, sep="\n")
            f.write("\n")
        return

    def write_vtu(self):
        with open(self._filename, "wt") as f:
            f.write(
                "<?xml version = \"1.0\"?>\n"
                "<VTKFile type = \"UnstructuredGrid\" version = \"0.1\" byte_order = \"LittleEndian\">\n"
                "<UnstructuredGrid>\n"
                "<Piece NumberOfPoints = \"{:d}\" NumberOfCells = \"{:d}\"> \n".format(self._nodes.shape[0], self._elements.shape[0])  )

            f.write(
                "<Points>\n"
                "<DataArray type = \"Float32\" Name = \"Points\" NumberOfComponents = \"3\" format = \"binary\">\n")
            data_bytes = self._nodes.astype(np.float32).tobytes()
            header = np.array(len(data_bytes), dtype=np.uint)
            f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

            f.write("\n"
                "</DataArray>\n"
                "</Points>\n"
                "<Cells>\n"
                "<DataArray type = \"Int32\" Name = \"connectivity\" format = \"binary\">\n")
            data_bytes = self._elements.astype(np.int32).tobytes()
            header = np.array(len(data_bytes), dtype=int)
            f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

            f.write("\n"
                "</DataArray>\n"
                "<DataArray type = \"Int32\" Name = \"offsets\" format = \"binary\">\n")
            offsets = np.arange(self._elements.shape[1], self._elements.size+1, self._elements.shape[1], dtype=np.int32)
            data_bytes = offsets.tobytes()
            header = np.array(len(data_bytes), dtype=int)
            f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

            f.write("\n"
                "</DataArray>\n"
                "<DataArray type = \"Int32\" Name = \"types\" format = \"binary\">\n")
            types = np.full(self._elements.shape[0], type).astype(np.int32)
            data_bytes = types.tobytes()
            header = np.array(len(data_bytes), dtype=int)
            f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

            f.write("\n"
                "</DataArray>\n"
                "</Cells>\n"
                "</Piece>\n"
                "</UnstructuredGrid>\n"
                "</VTKFile>\n")
        return
    
    def __call__(self):
        exp = self._filename.split(".")[-1]
        if (exp == "inp"):
            self.write_inp()
        elif (exp == "vtk"):
            self.write_vtk()
        elif (exp == "vtu"):
            self.write_vtu()
        return



if __name__=='__main__':
    print("# vtk DataFile Version 5.1\n"
        "Volume Mesh\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n")
    data_bytes = np.full(100,33333).tobytes()
    header = np.array(len(data_bytes), dtype=np.float64)
    print(int(33333).to_bytes(length=4,byteorder='little'))
    print(header.tobytes())
    print(base64.b64encode(header.tobytes() + data_bytes).decode())