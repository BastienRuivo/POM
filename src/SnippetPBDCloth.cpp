// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2022 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

// ****************************************************************************
// This snippet illustrates cloth simulation using position-based dynamics
// particle simulation. It creates a piece of cloth that drops onto a rotating
// sphere. 
// ****************************************************************************
#include <vector>
#include <iostream>

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"

#include "snippetrender/SnippetRender.h"
#include "snippetrender/SnippetCamera.h"

#define CUDA_SUCCESS 0
#define SHOW_SOLID_SDF_SLICE 0
#define IDX(i, j, k, offset) ((i) + dimX * ((j) + dimY * ((k) + dimZ * (offset))))


#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "snippetcommon/SnippetPrint.h"
#include "snippetcommon/SnippetPVD.h"
#include "snippetutils/SnippetUtils.h"
#include "extensions/PxParticleExt.h"

using namespace physx;
using namespace ExtGpu;

extern void initPhysics(bool interactive);
extern void stepPhysics(bool interactive);	
extern void cleanupPhysics(bool interactive);
extern void keyPress(unsigned char key, const PxTransform& camera);
extern PxPBDParticleSystem* getParticleSystem();
extern PxParticleClothBuffer* getUserClothBuffer();

namespace
{
Snippets::Camera* sCamera;

Snippets::SharedGLBuffer sPosBuffer;
PxU32 nbTriangle;
PxU32 nbParts;
PxVec4 * array;
PxU32 * ind;

unsigned int EBO;


void onBeforeRenderParticles()
{
	PxPBDParticleSystem* particleSystem = getParticleSystem();
	if (particleSystem) 
	{
		PxParticleClothBuffer* userBuffer = getUserClothBuffer();
		PxVec4* positions = userBuffer->getPositionInvMasses();
		PxU32* indexes = userBuffer->getTriangles();
		const PxU32 numParticles = userBuffer->getNbActiveParticles();
		nbParts = numParticles;

		PxScene* scene;
		PxGetPhysics().getScenes(&scene, 1);
		PxCudaContextManager* cudaContexManager = scene->getCudaContextManager();
		

		cudaContexManager->acquireContext();
		PxCudaContext* cudaContext = cudaContexManager->getCudaContext();
		array = new PxVec4[numParticles];
		ind = new PxU32[nbTriangle * 3];

		cudaContext->memcpyDtoH(sPosBuffer.map(), CUdeviceptr(positions), sizeof(PxVec4) * numParticles);
		cudaContext->memcpyDtoH(array, CUdeviceptr(positions), sizeof(PxVec4) * numParticles);
		cudaContext->memcpyDtoH(ind, CUdeviceptr(indexes), sizeof(PxU32) * nbTriangle * 3);


		


		cudaContexManager->releaseContext();
#if SHOW_SOLID_SDF_SLICE
		particleSystem->copySparseGridData(sSparseGridSolidSDFBufferD, PxSparseGridDataFlag::eGRIDCELL_SOLID_GRADIENT_AND_SDF);
#endif
	}
	
}



void renderParticles()
{
	
	sPosBuffer.unmap();
	
	PxVec3 color(1.f, 1.f, 1.f);
	
	Snippets::renderMesh(nbParts, array, nbTriangle, ind, PxVec3(1.0f, 1.f, 1.f));

	
	//Snippets::DrawPoints(sPosBuffer.vbo, sPosBuffer.size / sizeof(PxVec4), {1.f, 0.f, 0.f}, 2.f);
	Snippets::DrawFrame(PxVec3(0, 0, 0));
}



void allocParticleBuffers()
{
	PxScene* scene;
	PxGetPhysics().getScenes(&scene, 1);
	PxCudaContextManager* cudaContexManager = scene->getCudaContextManager();

	PxParticleClothBuffer* userBuffer = getUserClothBuffer();
	PxU32 maxParticles = userBuffer->getMaxParticles();

	sPosBuffer.initialize(cudaContexManager);
	sPosBuffer.allocate(maxParticles * sizeof(PxVec4));
}

void clearupParticleBuffers()
{
	sPosBuffer.release();
}

void renderCallback()
{
	onBeforeRenderParticles();

	stepPhysics(true);


	Snippets::startRender(sCamera);

	glDisable(GL_CULL_FACE); 
	PxScene* scene;
	PxGetPhysics().getScenes(&scene,1);
	PxU32 nbActors = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC);
	if(nbActors)
	{
		std::vector<PxRigidActor*> actors(nbActors);
		scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC, reinterpret_cast<PxActor**>(&actors[0]), nbActors);
		Snippets::renderActors(&actors[0], static_cast<PxU32>(actors.size()), true, PxVec3(1.f, 0.f, 0.f), NULL, true, false);
	}
	
	
	renderParticles();

	Snippets::showFPS();

	Snippets::finishRender();
}

void cleanup()
{
	delete sCamera;
	clearupParticleBuffers();
	cleanupPhysics(true);
}

void exitCallback(void)
{
#if PX_WINDOWS
	cleanup();
#endif
}
}

void renderLoop()
{
	sCamera = new Snippets::Camera(PxVec3(15.0f, 10.0f, 15.0f), PxVec3(-0.6f,-0.2f,-0.6f));

	Snippets::setupDefault("PhysX Snippet PBDCloth", sCamera, keyPress, renderCallback, exitCallback);

	initPhysics(true);
    glGenBuffers(1, &EBO);

	allocParticleBuffers();

	glutMainLoop();

#if PX_LINUX_FAMILY
	cleanup();
#endif
}

static PxDefaultAllocator			gAllocator;
static PxDefaultErrorCallback		gErrorCallback;
static PxFoundation*				gFoundation			= NULL;
static PxPhysics*					gPhysics			= NULL;
static PxDefaultCpuDispatcher*		gDispatcher			= NULL;
static PxScene*						gScene				= NULL;
static PxMaterial*					gMaterial			= NULL;
static PxPvd*						gPvd				= NULL;
static PxPBDParticleSystem*			gParticleSystem		= NULL;
static PxParticleClothBuffer*		gClothBuffer		= NULL;
static ExtGpu::PxParticleAttachmentBuffer *		gAttachementBuffer		= NULL;
static bool							gIsRunning			= true;

PxRigidDynamic* sphere;

static void initObstacles()
{
	PxShape* shape = gPhysics->createShape(PxSphereGeometry(3.0f), *gMaterial);
	// sphere = gPhysics->createRigidDynamic(PxTransform(PxVec3(0.f, 5.0f, 0.f)));
	// sphere->attachShape(*shape);
	// sphere->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
	// gScene->addActor(*sphere);
	shape->release();
}



// -----------------------------------------------------------------------------------------------------------------
static void initScene()
{
	PxCudaContextManager* cudaContextManager = NULL;
	if (PxGetSuggestedCudaDeviceOrdinal(gFoundation->getErrorCallback()) >= 0)
	{
		// initialize CUDA
		PxCudaContextManagerDesc cudaContextManagerDesc;
		cudaContextManager = PxCreateCudaContextManager(*gFoundation, cudaContextManagerDesc, PxGetProfilerCallback());
		if (cudaContextManager && !cudaContextManager->contextIsValid())
		{
			cudaContextManager->release();
			cudaContextManager = NULL;
		}
	}
	if (cudaContextManager == NULL)
	{
		PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION, __FILE__, __LINE__, "Failed to initialize CUDA!\n");
	}

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.cudaContextManager = cudaContextManager;
	sceneDesc.staticStructure = PxPruningStructureType::eDYNAMIC_AABB_TREE;
	sceneDesc.flags |= PxSceneFlag::eENABLE_PCM;
	sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
	sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
	sceneDesc.solverType = PxSolverType::eTGS;
	gScene = gPhysics->createScene(sceneDesc);
}

// -----------------------------------------------------------------------------------------------------------------
static PX_FORCE_INLINE PxU32 id(PxU32 x, PxU32 y, PxU32 numY)
{
	return x * numY + y;
}

static void initCloth(const PxU32 numX, const PxU32 numZ, const PxVec3& position = PxVec3(0, 10, 0), const PxReal particleSpacing = 0.2f, const PxReal totalClothMass = 10.f, PxRigidActor* toAttach = nullptr)
{
	PxCudaContextManager* cudaContextManager = gScene->getCudaContextManager();
	if (cudaContextManager == NULL)
		return;

	const PxU32 numParticles = numX * numZ;
	const PxU32 numSprings = (numX - 1) * (numZ - 1) * 4 + (numX - 1) + (numZ - 1);
	const PxU32 numTriangles = (numX - 1) * (numZ - 1) * 2;

	const PxReal restOffset = particleSpacing;
	
	const PxReal stretchStiffness = 10000.f;
	const PxReal shearStiffness = 100.f;
	const PxReal springDamping = 0.001f;

	// Material setup
	PxPBDMaterial* defaultMat = gPhysics->createPBDMaterial(0.8f, 0.05f, 1e+6f, 0.001f, 0.5f, 0.005f, 0.05f, 0.f, 0.f);

	PxPBDParticleSystem *particleSystem = gPhysics->createPBDParticleSystem(*cudaContextManager);
	gParticleSystem = particleSystem;

	// General particle system setting
	
	const PxReal particleMass = totalClothMass / numParticles;
	particleSystem->setRestOffset(restOffset);
	particleSystem->setContactOffset(restOffset + 0.02f);
	particleSystem->setParticleContactOffset(restOffset + 0.02f);
	particleSystem->setSolidRestOffset(restOffset);
	particleSystem->setFluidRestOffset(0.0f);

	gScene->addActor(*particleSystem);

	// Create particles and add them to the particle system
	const PxU32 particlePhase = particleSystem->createPhase(defaultMat, PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollideFilter | PxParticlePhaseFlag::eParticlePhaseSelfCollide));
	PxParticleClothBufferHelper* clothBuffers = PxCreateParticleClothBufferHelper(1, numTriangles, numSprings, numParticles, cudaContextManager);

	PxU32* phase = cudaContextManager->allocPinnedHostBuffer<PxU32>(numParticles);
	PxVec4* positionInvMass = cudaContextManager->allocPinnedHostBuffer<PxVec4>(numParticles);
	PxVec4* velocity = cudaContextManager->allocPinnedHostBuffer<PxVec4>(numParticles);
	
	PxReal x = position.x;
	PxReal y = position.y;
	PxReal z = position.z;
	std::cout<<position.z << " z = " << z<<std::endl;

	// Define springs and triangles
	
	PxArray<PxParticleSpring> springs;
	springs.reserve(numSprings);
	PxArray<PxU32> triangles;
	triangles.reserve(numTriangles * 3);

	for (PxU32 i = 0; i < numX; ++i)
	{
		for (PxU32 j = 0; j < numZ; ++j)
		{
			
			const PxU32 index = i * numZ + j;

			PxVec4 pos(x, y, z, 1.0f / particleMass);
			phase[index] = particlePhase;
			positionInvMass[index] = pos;
			velocity[index] = PxVec4(0.f, 0.f, 0.f, 0.f);
				
			if (i > 0)
			{
				PxParticleSpring spring = { id(i - 1, j, numZ), id(i, j, numZ), particleSpacing, stretchStiffness, springDamping, 0 };
				springs.pushBack(spring);
			}
			if (j > 0)
			{
				PxParticleSpring spring = { id(i, j - 1, numZ), id(i, j, numZ), particleSpacing, stretchStiffness, springDamping, 0 };
				springs.pushBack(spring);
			}
			
			if (i > 0 && j > 0) 
			{
				PxParticleSpring spring0 = { id(i - 1, j - 1, numZ), id(i, j, numZ), PxSqrt(2.0f) * particleSpacing, shearStiffness, springDamping, 0 };
				springs.pushBack(spring0);
				PxParticleSpring spring1 = { id(i - 1, j, numZ), id(i, j - 1, numZ), PxSqrt(2.0f) * particleSpacing, shearStiffness, springDamping, 0 };
				springs.pushBack(spring1);

				//Triangles are used to compute approximated aerodynamic forces for cloth falling down
				triangles.pushBack(id(i - 1, j - 1, numZ));
				triangles.pushBack(id(i - 1, j, numZ));
				triangles.pushBack(id(i, j - 1, numZ));

				triangles.pushBack(id(i - 1, j, numZ));
				triangles.pushBack(id(i, j, numZ));
				triangles.pushBack(id(i, j - 1, numZ));
			}

			y += particleSpacing;
		}
		y = position.y;
		x += particleSpacing;
	}

	PX_ASSERT(numSprings == springs.size());
	PX_ASSERT(numTriangles == triangles.size()/3);
	nbTriangle = numTriangles;
	
	clothBuffers->addCloth(0.0f, 0.0f, 0.0f, triangles.begin(), numTriangles, springs.begin(), numSprings, positionInvMass, numParticles);
	gParticleSystem->addRigidAttachment(toAttach);
	ExtGpu::PxParticleBufferDesc bufferDesc;
	bufferDesc.maxParticles = numParticles;
	bufferDesc.numActiveParticles = numParticles;
	bufferDesc.positions = positionInvMass;
	bufferDesc.velocities = velocity;
	bufferDesc.phases = phase;

	const PxParticleClothDesc& clothDesc = clothBuffers->getParticleClothDesc();
	PxParticleClothPreProcessor* clothPreProcessor = PxCreateParticleClothPreProcessor(cudaContextManager);

	PxPartitionedParticleCloth output;
	clothPreProcessor->partitionSprings(clothDesc, output);
	clothPreProcessor->release();

	gClothBuffer = physx::ExtGpu::PxCreateAndPopulateParticleClothBuffer(bufferDesc, clothDesc, output, cudaContextManager);
	gParticleSystem->addParticleBuffer(gClothBuffer);

	clothBuffers->release();

	cudaContextManager->freePinnedHostBuffer(positionInvMass);
	cudaContextManager->freePinnedHostBuffer(velocity);
	cudaContextManager->freePinnedHostBuffer(phase);
}

PxPBDParticleSystem* getParticleSystem()
{
	return gParticleSystem;
}

PxParticleClothBuffer* getUserClothBuffer()
{
	return gClothBuffer;
}

// -----------------------------------------------------------------------------------------------------------------
void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	gPvd = PxCreatePvd(*gFoundation);
	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
	gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);

	initScene();

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if (pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	// Setup rigid bodies
	const PxReal boxSize = 1.0f;
	const PxReal boxMass = 1.0f;
	float thickness = 0.01f;
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(thickness * boxSize, 5.f * boxSize, thickness * boxSize), *gMaterial);
	PxRigidStatic* body = gPhysics->createRigidStatic(PxTransform(PxVec3(-thickness/2, 5.0f, -thickness/2)));
	body->attachShape(*shape);
	
	
	gScene->addActor(*body);

	// Setup Cloth
	const PxReal totalClothMass = 10.0f;

	PxU32 numPointsX = 150;
	PxU32 numPointsZ = 100;
	PxReal particleSpacing = 0.05f;
	initCloth(numPointsX, numPointsZ, PxVec3(0.f, 5.f, 0.f), particleSpacing, totalClothMass, body);
	


	initObstacles();

	gScene->addActor(*PxCreatePlane(*gPhysics, PxPlane(0.f, 1.f, 0.f, 0.0f), *gMaterial));
	
	//gParticleSystem.add
	//gParticleSystem->setWind(PxVec3(0, 0, 0));

	shape->release();
}

// ---------------------------------------------------
PxReal simTime = 0;
void stepPhysics(bool /*interactive*/)
{
	if (gIsRunning)
	{
		const PxReal dt = 1.0f / 60.0f;

		// bool rotatingSphere = false;
		// if (rotatingSphere)
		// {
		// 	const PxReal speed = 2.0f;
		// 	PxTransform pose = sphere->getGlobalPose();			
		// 	sphere->setKinematicTarget(PxTransform(pose.p, PxQuat(PxCos(simTime*speed), PxVec3(0,1,0))));
		// }

		gScene->simulate(dt);
		gScene->fetchResults(true);
		gScene->fetchResultsParticleSystem();
		simTime += dt;
	}
}
	
void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		gPvd->release();	gPvd = NULL;
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	
	printf("SnippetPBDCloth done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	(void)camera;

	switch(toupper(key))
	{
	case 'P':	gIsRunning = !gIsRunning;	break;
	}
}

int main(int, const char*const*)
{
	renderLoop();

	return 0;
}
