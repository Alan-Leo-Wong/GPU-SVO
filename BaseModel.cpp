#include "BaseModel.h"

void BaseModel::loadOBJ(const std::string& in_file)
{
	std::ifstream in(in_file);
	if (!in)
	{
		std::cerr << "ERROR: loading obj:(" << in_file << ") file is not good" << std::endl;
		exit(1);
	}

	float x, y, z;
	int f0, _f0, f1, _f1, f2, _f2;
	char buffer[256] = { 0 };
	while (!in.getline(buffer, 255).eof())
	{
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			if (sscanf_s(buffer, "v %f %f %f", &x, &y, &z) == 3)
				modelPoints.emplace_back(Eigen::Vector3f{ x, y, z });
		}
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			if (sscanf_s(buffer, "f %d//%d %d//%d %d//%d", &f0, &_f0, &f1, &_f1, &f2, &_f2) == 6)
				idx2Points.emplace_back(Eigen::Vector3i{ f0 - 1, f1 - 1, f2 - 1 });
			else if (sscanf_s(buffer, "f %d/%d %d/%d %d/%d", &f0, &_f0, &f1, &_f1, &f2, &_f2) == 6)
				idx2Points.emplace_back(Eigen::Vector3i{ f0 - 1, f1 - 1, f2 - 1 });
			else if (sscanf_s(buffer, "f %d %d %d", &f0, &f1, &f2) == 3)
				idx2Points.emplace_back(Eigen::Vector3i{ f0 - 1, f1 - 1, f2 - 1 });
		}
	}
}