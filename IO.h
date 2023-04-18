#pragma once
#include <fstream>
#include <Eigen\Dense>
#include "MortonLUT.h"

// Helper function to write single vertex to OBJ file
static void write_vertex(std::ofstream& output, const Eigen::Vector3f& v) {
	output << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
}

// Helper function to write single vertex
static void write_face(std::ofstream& output, const Eigen::Vector3i& f) {
	output << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
}

// Helper function to write full cube (using relative vertex positions in the OBJ file - support for this should be widespread by now)
void write_cube(const Eigen::Vector3f& bboxOrigin, const Eigen::Vector3f& unit,
	const size_t& x, const size_t& y, const size_t& z, std::ofstream& output, size_t& nFaces) {
	//	   2-------1
	//	  /|      /|
	//	 / |     / |
	//	7--|----8  |
	//	|  4----|--3
	//	| /     | /
	//	5-------6
	// Create vertices
	Eigen::Vector3f v1 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x + 1, y + 1, z + 1) * Eigen::Array3f(unit));
	Eigen::Vector3f v2 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x, y + 1, z + 1) * Eigen::Array3f(unit));
	Eigen::Vector3f v3 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x + 1, y, z + 1) * Eigen::Array3f(unit));
	Eigen::Vector3f v4 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x, y, z + 1) * Eigen::Array3f(unit));
	Eigen::Vector3f v5 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x, y, z) * Eigen::Array3f(unit));
	Eigen::Vector3f v6 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x + 1, y, z) * Eigen::Array3f(unit));
	Eigen::Vector3f v7 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x, y + 1, z) * Eigen::Array3f(unit));
	Eigen::Vector3f v8 = bboxOrigin + Eigen::Vector3f(Eigen::Array3f(x + 1, y + 1, z) * Eigen::Array3f(unit));

	// write them in reverse order, so relative position is -i for v_i
	write_vertex(output, v8);
	write_vertex(output, v7);
	write_vertex(output, v6);
	write_vertex(output, v5);
	write_vertex(output, v4);
	write_vertex(output, v3);
	write_vertex(output, v2);
	write_vertex(output, v1);

	// create faces
	// back
	write_face(output, Eigen::Vector3i(nFaces + 1, nFaces + 3, nFaces + 4));
	write_face(output, Eigen::Vector3i(nFaces + 1, nFaces + 4, nFaces + 2));
	// bottom								   
	write_face(output, Eigen::Vector3i(nFaces + 4, nFaces + 3, nFaces + 6));
	write_face(output, Eigen::Vector3i(nFaces + 4, nFaces + 6, nFaces + 5));
	// right								   	 
	write_face(output, Eigen::Vector3i(nFaces + 3, nFaces + 1, nFaces + 8));
	write_face(output, Eigen::Vector3i(nFaces + 3, nFaces + 8, nFaces + 6));
	// top									   	 
	write_face(output, Eigen::Vector3i(nFaces + 1, nFaces + 2, nFaces + 7));
	write_face(output, Eigen::Vector3i(nFaces + 1, nFaces + 7, nFaces + 8));
	// left									   	 
	write_face(output, Eigen::Vector3i(nFaces + 2, nFaces + 4, nFaces + 5));
	write_face(output, Eigen::Vector3i(nFaces + 2, nFaces + 5, nFaces + 7));
	// front								   	  
	write_face(output, Eigen::Vector3i(nFaces + 5, nFaces + 6, nFaces + 8));
	write_face(output, Eigen::Vector3i(nFaces + 5, nFaces + 8, nFaces + 7));

	nFaces += 12;
}

void write_obj_cubes(const unsigned int* vtable, const Eigen::Vector3f& bboxOrigin,
	const Eigen::Vector3f& unit, const Eigen::Vector3i& gridsize, const std::string& base_filename)
{
	std::string filename_output = base_filename + std::string("_") + std::to_string(gridsize.x()) + std::string("_voxels.obj");
	std::ofstream output(filename_output.c_str(), std::ios::out);

#ifndef SILENT
	fprintf(stdout, "[I/O] Writing data in obj voxels format to file %s \n", filename_output.c_str());
	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = (size_t(gridsize.x()) * size_t(gridsize.y()) * size_t(gridsize.z())) / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	assert(output);
	size_t nFaces = 0;
	for (size_t x = 0; x < gridsize.x(); x++) {
		for (size_t y = 0; y < gridsize.y(); y++) {
			for (size_t z = 0; z < gridsize.z(); z++) {
#ifndef SILENT
				voxels_seen++;
				if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
				else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
				else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
#endif
				if (checkVoxel(x, y, z, gridsize, vtable)) {
					//voxels_written += 1;
					write_cube(bboxOrigin, unit, x, y, z, output, nFaces);
				}
			}
		}
	}
#ifndef SILENT
	fprintf(stdout, "100%% \n");
#endif
	// std::cout << "written " << voxels_written << std::endl;

	output.close();
}