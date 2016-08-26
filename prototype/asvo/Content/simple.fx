/*
Original code by Riemer Grootjans
*/

Texture colorTexture;
sampler colorTextureSampler = sampler_state { 
	texture = <colorTexture> ; magfilter = POINT; minfilter = POINT; mipfilter=POINT;
	AddressU = CLAMP; AddressV = CLAMP;
};

Texture depthTexture;
sampler depthTextureSampler = sampler_state { 
	texture = <depthTexture> ; magfilter = POINT; minfilter = POINT; mipfilter=POINT;
	AddressU = CLAMP; AddressV = CLAMP;
};

float pixelWidth;
float pixelHeight;
float4 fillColor;
float maxDim;
int horRes, vertRes;

struct VertexToPixel
{
    float4 Position  : POSITION0;    
    float2 TexCoords : TEXCOORD0;
};

struct PixelToFrame
{
    float4 Color : COLOR0;
};


VertexToPixel SimplestVertexShader(float4 inPos : POSITION0, float2 inTexCoords : TEXCOORD0)
{
    VertexToPixel Output = (VertexToPixel)0;
    
    Output.Position = inPos;
    Output.TexCoords = inTexCoords;

    return Output;
}

PixelToFrame OurFirstPixelShader(VertexToPixel PSIn)
{
    PixelToFrame Output = (PixelToFrame)0;
    
	int radius = 1;
	const int radiusTimes2 = radius * 2;
		
	float min = 1.0f;
	Output.Color = fillColor;
	const float uStart = PSIn.TexCoords.x - radius * pixelWidth;
	float u = uStart;
	float v = PSIn.TexCoords.y - radius * pixelHeight;		
	float uMin, vMin;		
	
	for (int i = 0; i <= radiusTimes2; ++i)
	{
		for (int j = 0; j <= radiusTimes2; ++j)
		{
			float depth = tex2D(depthTextureSampler, float2(u, v)).r;
			if (depth < min)
			{			
				min = depth;
				uMin = u;
				vMin = v;
			}
			u += pixelWidth;
		}
		u = uStart;
		v += pixelHeight;
	}
	
	if (min != 1.0f)
		Output.Color = tex2D(colorTextureSampler, float2(uMin, vMin));
	   
    return Output;
}

technique Simplest
{
    pass Pass0
    {
        VertexShader = compile vs_3_0 SimplestVertexShader();
        PixelShader = compile ps_3_0 OurFirstPixelShader();
    }
}