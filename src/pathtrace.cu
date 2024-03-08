#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <windows.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	volatile float p1 = 1, p2 = 1, p3 = 1;
	volatile float c1 = 1, c2 = 1, c3 = 1;
	int index = x + (y * resolution.x);
	if (x < resolution.x && y < resolution.y) {
		glm::vec3 pix = image[index];

		glm::vec3 color;
		p1 = pix.x, p2 = pix.y, p3 = pix.z;
#if TONEMAPPING
		color = pix / (float)iter;
		c1 = color.x, c2 = color.y, c3 = color.z;
		color = gammaCorrection(ACESFilm(color));
		c1 = color.x, c2 = color.y, c3 = color.z;
		color = glm::clamp(color * 255.0f, glm::vec3(0.f), glm::vec3(255.0f));
		c1 = color.x, c2 = color.y, c3 = color.z;
#else
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);
#endif


		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths1 = NULL;
static PathSegment* dev_paths2 = NULL;
static ShadeableIntersection* dev_intersections1 = NULL;
static ShadeableIntersection* dev_intersections2 = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static DevScene* dev_scene = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	dev_scene = hst_scene->dev_scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths1, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections1, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections2, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaSafeFree(dev_image);  // no-op if dev_image is null
	cudaSafeFree(dev_paths1);
	cudaSafeFree(dev_paths2);
	cudaSafeFree(dev_geoms);
	cudaSafeFree(dev_materials);
	cudaSafeFree(dev_intersections1);
	cudaSafeFree(dev_intersections2);
	// TODO: clean up any extra device memory you created

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	volatile float r1 = 1, r2 = 1, d1 = 1, d2 = 1, d3 = 1, v1 = 1, v2 = 1, v3 = 1, ri1 = 1, ri2 = 1, ri3 = 1;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		Sampler rng = makeSeededRandomEngine(x, y, iter);
		glm::vec2 r = sample2D(rng);
		
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * static_cast<float>(static_cast<float>(x) + (r.x - 0.5) - cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * static_cast<float>(static_cast<float>(y) + (r.y - 0.5) - cam.resolution.y * 0.5f));
		r1 = r.x, r2 = r.y, d1 = segment.ray.direction.x, d2 = segment.ray.direction.y, d3 = segment.ray.direction.z;
		v1 = cam.view.x, v2 = cam.view.y, v3 = cam.view.z, ri1 = cam.right.x, ri2 = cam.right.y, ri3 = cam.right.z;

		segment.pixelIndex = index;

		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int iter,
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, DevScene* dev_scene
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* img
	, Material* materials
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	volatile float n1 = 1, n2 = 1, n3 = 1, c1 = 1, c2 = 1, c3 = 1, e1 = 1, e2 = 1, e3 = 1;
	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t = -1;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 texCoords;
		glm::vec3 T, B;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		int triangleId = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		GpuBVHNode* gpuBVH = dev_scene->dev_gpuBVH;
		Triangle* triangles = dev_scene->dev_triangles;

		volatile float p1, p2, p3;

		//naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				p1 = tmp_intersect.x, p2 = tmp_intersect.y, p3 = tmp_intersect.z;
				normal = tmp_normal;
			}
		}
#if USE_BVH
		int bvhIdx = 0;
		Triangle tempTri;
		float tempT = FLT_MAX;
		int offset = 0;
#if USE_MTBVH
		glm::vec3 dir = pathSegment.ray.direction;
		offset = ((abs(dir[0]) > abs(dir[1])) && (abs(dir[0]) > abs(dir[2]))) ? 0 : (abs(dir[1]) > abs(dir[2]) ? 1 : 2);
		offset = offset + (dir[offset] > 0 ? 0 : 3);
		offset *= dev_scene->bvh_size;
#endif
		GpuBVHNode* curBVH = gpuBVH + offset;
		volatile int count = 0;
		while (bvhIdx != -1)
		{
			count++;
			if (!(curBVH[bvhIdx].bBox.IntersectP(pathSegment.ray, tempT)) || tempT > t_min)
			{
				bvhIdx = curBVH[bvhIdx].miss;
				continue;
			}
			//it indicates gpuBVH[bvhIdx] is a leaf node
			if (curBVH[bvhIdx].end - curBVH[bvhIdx].start <= MAX_PRIM)
			{
				for (int i = curBVH[bvhIdx].start; i < curBVH[bvhIdx].end; ++i)
				{
					tempTri = triangles[i];
					float u, v;
					bool isHit = tempTri.getInterSect(pathSegment.ray, t, u, v);
					if (isHit && t_min > t)
					{
						t_min = t;
						hit_geom_index = tempTri.geomIdx;
						triangleId = i;
						intersect_point = pathSegment.ray.origin + t * pathSegment.ray.direction;
						intersect_point = (1 - u - v) * tempTri.v[0] + u * tempTri.v[1] + v * tempTri.v[2];
						normal = (1 - u - v) * tempTri.n[0] + u * tempTri.n[1] + v * tempTri.n[2];
						texCoords = (1 - u - v) * tempTri.tex[0] + u * tempTri.tex[1] + v * tempTri.tex[2];
						T = tempTri.tangent, B = tempTri.bitangent;
					}
				}
			}
			bvhIdx = curBVH[bvhIdx].hit;
		}
#else
		for (int i = 0; i < dev_scene->tri_num; ++i)
		{
			float u, v;
			Triangle tempTri = dev_scene->dev_triangles[i];
			bool isHit = tempTri.getInterSect(pathSegment.ray, t, u, v);
			if (isHit && t_min > t)
			{
				t_min = t;
				//MYTODO
				hit_geom_index = tempTri.geomIdx;
				intersect_point = pathSegment.ray.origin + t * pathSegment.ray.direction;
				intersect_point = (1 - u - v) * tempTri.v[0] + u * tempTri.v[1] + v * tempTri.v[2];
				normal = (1 - u - v) * tempTri.n[0] + u * tempTri.n[1] + v * tempTri.n[2];
			}
		}
#endif

#if SHOW_NORMAL
		intersections[path_index].t = -1.0f;
		rayValid[path_index] = 0;
		img[pathSegments[path_index].pixelIndex] += math::processNAN(glm::normalize(normal) + glm::vec3(1.f));
#else
		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			rayValid[path_index] = 0;
			if (dev_scene->envMapID >= 0)
			{
				img[pathSegments[path_index].pixelIndex] += math::processNAN(pathSegments[path_index].color * dev_scene->envSampler.linearSample(math::sphere2Plane(pathSegment.ray.direction)));
			}
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].interPoint = intersect_point;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].texCoords = glm::clamp(texCoords, 0.f, 1.f);
			intersections[path_index].geomID = hit_geom_index;
			intersections[path_index].triangleID = triangleId;
			rayValid[path_index] = 1;

			glm::vec3 mapped = materials[geoms[hit_geom_index].materialid].normalSampler.linearSample(texCoords);
			glm::vec3 localNorm = glm::normalize(glm::vec3(mapped.x, mapped.y, mapped.z) * 1.f - 0.5f);
			normal = glm::normalize(normal);

			//intersections[path_index].surfaceNormal = math::localRefMatrix2(normal) * localNorm;
			// 
			//intersections[path_index].surfaceNormal = normal;

			if (glm::dot(T, T) > 1e-3f && glm::abs(localNorm.z - 1.f) > 1e-5)
			{
				B = glm::normalize(glm::cross(normal, T));
				T = glm::normalize(glm::cross(B, normal));
				intersections[path_index].surfaceNormal = glm::normalize(glm::mat3(T, B, normal) * localNorm);
			}
			else
			{
				intersections[path_index].surfaceNormal = normal;
			}

			n1 = mapped.x, n2 = mapped.y, n3 = mapped.z;
		}
#endif // SHOW_NORMAL
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void PTkernel(
	int iter
	, int depth
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int *rayValid
	, glm::vec3 *img
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
	{
		return;
	}
	volatile int textID = pathSegments[idx].pixelIndex;
	volatile float c01 = pathSegments[idx].color.x, c02 = pathSegments[idx].color.y, c03 = pathSegments[idx].color.z;
	volatile float o1 = pathSegments[idx].ray.origin.x, o2 = pathSegments[idx].ray.origin.y, o3 = pathSegments[idx].ray.origin.z;
	volatile float d1 = pathSegments[idx].ray.direction.x, d2 = pathSegments[idx].ray.direction.y, d3 = pathSegments[idx].ray.direction.z;
	volatile float n1 = shadeableIntersections[idx].surfaceNormal.x, n2 = shadeableIntersections[idx].surfaceNormal.y, n3 = shadeableIntersections[idx].surfaceNormal.z;
	volatile float p1 = shadeableIntersections[idx].interPoint.x, p2 = shadeableIntersections[idx].interPoint.y, p3 = shadeableIntersections[idx].interPoint.z;
	volatile int mid = shadeableIntersections[idx].materialId;
	volatile float ic1 = img[pathSegments[idx].pixelIndex].x, ic2 = img[pathSegments[idx].pixelIndex].y, ic3 = img[pathSegments[idx].pixelIndex].z;
	volatile float b1 = 1, b2 = 1, b3 = 1, tmppdf = 1;
	//volatile float off1 = p1, off2 = p2, off3 = p3, testt = shadeableIntersections[idx].t;

	ShadeableIntersection intersection = shadeableIntersections[idx];
	glm::vec3 wo = -pathSegments[idx].ray.direction;
	Material::Type mType = materials[intersection.materialId].type;

	// If there was no intersection, color the ray black.
	// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
	// used for opacity, in which case they can indicate "no opacity".
	// This can be useful for post-processing and image compositing.
	if (intersection.t <= 0.f)
	{
		pathSegments[idx].color *= 0;
		pathSegments[idx].remainingBounces = 0;
		rayValid[idx] = 0;
		return;
	}

	// if the intersection exists...
	// Set up the RNG
	// LOOK: this is how you use thrust's RNG! Please look at
	// makeSeededRandomEngine as well.

	//if (mType != Material::Type::Dielectric && mType != Material::Type::Light && glm::dot(wo, intersection.surfaceNormal) < 0)
	//{
	//	intersection.surfaceNormal = -intersection.surfaceNormal;
	//}

	Sampler rng = makeSeededRandomEngine(iter, idx, depth);
	scatter_record srec;
	bool ifScatter = materials[intersection.materialId].scatterSample(intersection, pathSegments[idx].ray.direction, srec, rng);

	if (srec.pdf == 0)
	{
		pathSegments[idx].color *= 0;
		pathSegments[idx].remainingBounces = 0;
		rayValid[idx] = 0;
	}
	// If the material indicates that the object is a light
	else if (mType == Material::Type::Light) {
		pathSegments[idx].color *= (srec.bsdf / srec.pdf);
		pathSegments[idx].remainingBounces = 0;
		rayValid[idx] = 0;
		img[pathSegments[idx].pixelIndex] += math::processNAN(pathSegments[idx].color);
	}

	else {
		glm::vec3 offsetDir = glm::dot(srec.dir, intersection.surfaceNormal) > 0 ? intersection.surfaceNormal : -intersection.surfaceNormal;
		pathSegments[idx].ray.direction = srec.dir;
		//pathSegments[idx].ray.origin = intersection.interPoint + 
		//								(mType == Material::Type::Dielectric ? 100 : 1) * RAY_BIAS * offsetDir;
		pathSegments[idx].ray.origin = intersection.interPoint + (mType == Material::Type::Dielectric ? 1e-3f * offsetDir : 1e-4f * srec.dir);
		pathSegments[idx].color *= (srec.bsdf * glm::abs(glm::dot(srec.dir, intersection.surfaceNormal)) / srec.pdf);
		rayValid[idx] = 1;

		b1 = srec.bsdf.x, b2 = srec.bsdf.y, b3 = srec.bsdf.z, tmppdf = srec.pdf;
		d1 = glm::abs(glm::dot(srec.dir, intersection.surfaceNormal));
		d1 = srec.dir.x, d2 = srec.dir.y, d3 = srec.dir.z;
		ic1 = offsetDir.x, ic2 = offsetDir.y, ic3 = offsetDir.z;
		o1 = pathSegments[idx].ray.origin.x, o2 = pathSegments[idx].ray.origin.y, o3 = pathSegments[idx].ray.origin.z;
		c01 = pathSegments[idx].color.x, c02 = pathSegments[idx].color.y, c03 = pathSegments[idx].color.z;

		if (--(pathSegments[idx].remainingBounces) == 0)
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			rayValid[idx] = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	volatile int textID = iterationPaths[index].pixelIndex;
	volatile float c1, c2, c3;
	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		c1 = iterationPath.color.x, c2 = iterationPath.color.y, c3 = iterationPath.color.z;
		image[iterationPath.pixelIndex] += iterationPath.color;		
	}
}

struct remain_bounce
{
	__host__ __device__
		bool operator()(const PathSegment& pathSegment)
	{
		return pathSegment.remainingBounces > 0;
	}
};

// from Andrew Yang https://github.com/bdwhst/Project3-CUDA-Path-Tracer
int compact_rays(int* rayValid, int* rayIndex, int numRays)
{
	if (numRays == 0) return 0;
	thrust::device_ptr<PathSegment> dev_thrust_paths1(dev_paths1), dev_thrust_paths2(dev_paths2);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections1(dev_intersections1), dev_thrust_intersections2(dev_intersections2);
	thrust::device_ptr<int> dev_thrust_rayValid(rayValid), dev_thrust_rayIndex(rayIndex);
	thrust::exclusive_scan(dev_thrust_rayValid, dev_thrust_rayValid + numRays, dev_thrust_rayIndex);
	int nextNumRays, tmp;
	cudaMemcpy(&tmp, rayIndex + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays = tmp;
	cudaMemcpy(&tmp, rayValid + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays += tmp;
	thrust::scatter_if(dev_thrust_paths1, dev_thrust_paths1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_paths2);
	thrust::scatter_if(dev_thrust_intersections1, dev_thrust_intersections1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_intersections2);
	std::swap(dev_paths1, dev_paths2);
	std::swap(dev_intersections1, dev_intersections2);
	return nextNumRays;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths1);
	checkCUDAError("generate camera ray");

	int* rayValid, *rayIndex;
	cudaMalloc(&rayValid, sizeof(int) * pixelcount);
	cudaMalloc(&rayIndex, sizeof(int) * pixelcount);

	int depth = 0;
	PathSegment* dev_path_end = dev_paths1 + pixelcount;
	int num_paths = dev_path_end - dev_paths1;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	LARGE_INTEGER t1, t2, tc;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		//QueryPerformanceFrequency(&tc);
		//QueryPerformanceCounter(&t1);

		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter
			, depth
			, num_paths
			, dev_paths1
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_scene
			, dev_intersections1
			, rayValid
			, dev_image
			, dev_materials
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		//QueryPerformanceCounter(&t2);
		//double time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
		//cout << "time1 = " << time << endl;

		num_paths = compact_rays(rayValid, rayIndex, num_paths);

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		//QueryPerformanceCounter(&t1);
		PTkernel << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth,
			num_paths,
			dev_intersections1,
			dev_paths1,
			dev_materials,
			rayValid,
			dev_image
			);
		//QueryPerformanceCounter(&t2);
		//time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
		//cout << "time2 = " << time << endl;

		num_paths = compact_rays(rayValid, rayIndex, num_paths);
		iterationComplete = (num_paths == 0); // TODO: should be based off stream compaction results.
		depth++;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	//dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths1);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	cudaFree(rayValid);
	cudaFree(rayIndex);

	checkCUDAError("pathtrace");
}
