#include "VisualData.h"

void VisualData::deserialize( FILE * inFile )
{
	fread( & normal,      12, 1, inFile );
	fread( & tangent,     12, 1, inFile );
	fread( & texCoord,     8, 1, inFile );
	fread( & boneIndex0,   1, 1, inFile );
	fread( & boneIndex1,   1, 1, inFile );
	fread( & boneIndex2,   1, 1, inFile );
	fread( & boneIndex3,   1, 1, inFile );
	fread( & boneWeights, 16, 1, inFile );
}