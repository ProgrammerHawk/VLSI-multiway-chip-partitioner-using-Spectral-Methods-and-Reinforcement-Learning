#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <assert.h>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <random>
#include <limits>
#include <float.h>
#include <sys/stat.h>
#include "kmeans.h"
#include "json.h"

namespace SpectralSolver {

class XY;
class Net;
class NLInst;
class FpgaCell;
class NetlistGraph;
class FpgaPairLink;
class MHSolver;
class MHMove;
class MHOpto;
class AnalyticalPartitioner;


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::SparseMatrix;

using NetNameMap = std::unordered_map<std::string, Net>;
using CellInstMap = std::unordered_map<std::string, NLInst>;
using NetNameVec = std::vector<std::string>; // For ordered nets by id
using CellInstNameVec = std::vector<std::string>; // For ordered cell instances by id
using NetIdVec = std::vector<unsigned>;
using CellInstIdVec = std::vector<unsigned>;
using NetVec = std::vector<Net>;
using CellInstVec = std::vector<NLInst>;
using FpgaCellIdVec = std::vector<unsigned>;
using FpgaCellVec = std::vector<FpgaCell>;
using SourceDestInstPair = std::pair<unsigned, unsigned>;
using SourceDestFpgaPair = std::pair<unsigned, unsigned>;
using LinkIdVec = std::vector<unsigned>;
using LinkVec = std::vector<FpgaPairLink>;
using InstIdSet = std::unordered_set<unsigned>;
using NetData = std::tuple<std::string, SourceDestInstPair, unsigned>; //name, Inst pair, weight
// name, incident nets, luts, bmems, dmems
using InstData = std::tuple<std::string, std::vector<unsigned>, unsigned, unsigned, unsigned>;
using InstDataVec = std::vector<InstData>;
using InstVec = std::vector<unsigned>;
using NetDataVec = std::vector<NetData>;
using LinkData = std::tuple<unsigned, unsigned, unsigned, bool>;
using LinkDataMap = std::map<unsigned, std::vector<LinkData>>;
using CellData = std::tuple<unsigned, unsigned, unsigned>; // Luts, BMems, DMems
using CellDataVec = std::vector<CellData>;
using SysDefData = std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned>;
using XYVec = std::vector<XY>;
using SimVec = std::vector<double>; // Cosine similarity vec
using FpgaSeedInsts = std::vector<unsigned>;
using CellActive2OrigIds = std::vector<unsigned>;
using ClusterCentroid2Insts = std::unordered_map<unsigned, std::set<unsigned>>;
using ClusterSetVec = std::vector<std::unordered_set<unsigned>>;
using FpgaProps = std::vector<std::tuple<bool, unsigned, unsigned>>;
using InstProps = std::vector<std::tuple<bool, bool, bool, unsigned>>;
using FpgaBridgeMtx = std::vector<std::vector<unsigned>>;
using PartNumInstsVec = std::vector<unsigned>;
using InstToPartVec = std::vector<unsigned>;
using PartToInstsSV = std::vector<std::unordered_set<unsigned>>;
using InstMoves = std::vector<std::tuple<unsigned, unsigned>>; // P1,P2
using InstMoveCosts = std::vector<double>;
using BBox = std::tuple<XY, XY, XY, XY>; // ll, lr, ul, ur
using FpgaBBoxVec = std::vector<BBox>; 

using json = nlohmann::json;

constexpr size_t SIZET_MAX = std::numeric_limits<std::size_t>::max();
constexpr float  MFLT_MIN = std::numeric_limits<float>::lowest();
constexpr double MDBL_MIN = std::numeric_limits<double>::lowest();

enum SysPhysAttr {
    SPA_FPGA,
    SPA_LUTS,
    SPA_ULUTS,
    SPA_BMEM,
    SPA_UBMEM,
    SPA_DMEM,
    SPA_UDMEM,
    SPA_LINK_DATA,
    SPA_BW,
    SPA_UNCONN,
    SPA_LINK_WT,
    SPA_MAX
};

enum InstGraphAttr {
    IGA_NUM_INSTS,
    IGA_NUM_NETS,
    IGA_TOTAL_DNETS,
    IGA_HAS_BMEM,
    IGA_HAS_DMEM,
    IGA_INST_NET_MAP,
    IGA_INST_NAMES,
    IGA_ALL_INSTS,
    IGA_ALL_NETS,
    IGA_INST_LUT_COUNT,
    IGA_INST_BMEM_COUNT,
    IGA_INST_DMEM_COUNT,
    IGA_EDGE_NETS_COUNT,
    IGA_MAX
};

enum CostFuncs {
    WWL_PHY_COST,
    WWL_RT_PHY_COST,
    RT_PHY_COST
};

struct RankComp
{
    template<typename T>
    bool operator()(const T& l, const T& r) const
    {
        return l.second > r.second;
    }
};

auto RandomNumberBetween = [](int low, int high)
{
    auto randomFunc = [distribution_ = std::uniform_int_distribution<int>(low, high), 
                       random_engine_ = std::mt19937{ std::random_device{}() }]() mutable
    {
        return distribution_(random_engine_);
    };
    return randomFunc;
};

using FpgaNeighborSet = std::multiset<std::pair<unsigned, float>, RankComp>;
using FpgaNeighbors = std::unordered_map<unsigned, FpgaNeighborSet>;

template <typename T>
class Matrix2D
{
    public:
        Matrix2D ()
            :sizeX(1), sizeY(1)
        {
            data.resize(sizeX*sizeY);
        }
        Matrix2D (unsigned x, unsigned y)
            : sizeX(x), sizeY(y)
        {
            data.clear();
            data.resize(sizeX*sizeY);
        }

        Matrix2D (const Matrix2D& other)
        {
            data.clear();
            sizeX = other.getXSize();
            sizeY = other.getYSize();
            data = other.getData();
        }

        void fill(const T& val)
        {
            data.clear();
            data.resize(sizeX*sizeY, val);
        }

        void assign(const std::vector<T>& valVec)
        {
            assert(valVec.size() == data.size());
            data = valVec;
        }

        Matrix2D& operator=(const Matrix2D& other) = default;
        
        unsigned getXSize() const
        {
            return sizeX;
        }
        unsigned getYSize() const
        {
            return sizeY;
        }
        const std::vector<T>& getData() const
        {
            return data;
        }
        void operator()(unsigned x, unsigned y, const T& val)
        {
            if (x >= sizeX || y >= sizeY)
                throw std::out_of_range("Bad matrix access");
            data[sizeY*x + y] = val; // Stride-aware access
            
        }

        const T& operator()(unsigned x, unsigned y) const
        {
            if (x >= sizeX || y >= sizeY)
                throw std::out_of_range("Bad matrix access");
            return data[sizeY*x + y]; // Stride-aware access
        }
        
    private:
        std::vector<T> data;
        unsigned sizeX, sizeY;
};

class XY 
{
    public:
        XY() : mX(0), mY(0) {}
        XY(float x, float y) : mX(x), mY(y) {}
        double getL2Norm() const
        {
            return std::sqrt((mX*mX) + (mY*mY));
        }
        double getAntiOrtho(const XY& other) const
        {
            double dot = mX * other.X() + mY* other.Y();
            double mag = getL2Norm() * other.getL2Norm();
            mag = mag > 0 ? mag : 1;
            return (dot/mag);
        }
        void getAntiOrthoVec(const XYVec& other, SimVec& sV) const
        {
            for (auto& oCd : other) {
                sV.emplace_back(getAntiOrtho(oCd));
            }
        }

        double getEDist(const XY& other) const
        {
            double xdiff = std::abs(other.X() - mX);
            double ydiff = std::abs(other.Y() - mY);
            return sqrt((xdiff*xdiff) + (ydiff*ydiff));
        }

        float X() const
        {
            return mX;
        }
        float Y() const
        {
            return mY;
        }
    private:
        float mX;
        float mY;
};

class Net
{
    // Net is a two pin edge in the graph
    public:
        Net(unsigned idx)
        {
            mNetId = idx;
            mNetWt = 0;
            mTotalInstsWt = 0;
            mIsCut = false;

        };
        ~Net() {}

        void setNetId(unsigned idx)
        {
            assert(idx >= 0);
            mNetId = idx;
        }
        void setSrcDestInstId(unsigned srcId, unsigned destId)
        {
            assert(srcId >= 0 && destId >= 0);
            mSrcDestInst = std::make_pair(srcId, destId);

        }
        void setNetWeight(unsigned wt)
        {
            mNetWt = wt;
        }
        void setNetIsCut(bool val)
        {
            mIsCut = val;
        }

        unsigned getNetId() const
        {
            return mNetId;
        }
        unsigned getNetSrc() const
        {
            return mSrcDestInst.first;
        }
        unsigned getNetDest() const
        {
            return mSrcDestInst.second;
        }
        const SourceDestInstPair& getNetSrcDest() const
        {
            return mSrcDestInst;
        }
        unsigned getWeight() const
        {
            return mNetWt;
        }
        bool getNetIsCut() const
        {
            return mIsCut;
        }
        unsigned getNetOtherInst(unsigned instId) const
        {
            assert(instId >= 0);
            if (mSrcDestInst.first == instId)
                return mSrcDestInst.second;
            else if (mSrcDestInst.second == instId)
                return mSrcDestInst.first;
            return instId; // Return the same
        }
    private:
        unsigned            mNetId;
        SourceDestInstPair  mSrcDestInst; // Source and Destination instance of net
        unsigned            mNetWt;
        size_t              mTotalInstsWt;
        bool                mIsCut;
};

class NLInst
{
    public:
        NLInst(unsigned id)
        {
            assert(id >= 0);
            mInstId = id;
            mLutWt = 0;
            mBMemWt = 0;
            mDMemWt = 0;
            mMasterId = id;
            mTotalNetWt = 0;
            mTotalNeighborWt = 0;
            mIsNeighborFixed = false;
            mIsFixed = false;
            mCurrPartition = UINT_MAX;
        }
        ~NLInst() {}

        unsigned getInstId() const
        {
            return mInstId;
        }
        void setLutWt(unsigned wt)
        {
            mLutWt = wt;
        }
        void setBMemWt(unsigned wt)
        {
            mBMemWt = wt;
        }
        void setDMemWt(unsigned wt)
        {
            mDMemWt = wt;
        }
        unsigned getLutWeight() const
        {
            return mLutWt;
        }
        unsigned getBMemWeight() const
        {
            return mBMemWt;
        }
        unsigned getDMemWeight() const
        {
            return mDMemWt;
        }
        size_t getTotalWeight() const
        {
            return (mLutWt + mBMemWt + mDMemWt);
        }
        void setTotalNetWt(size_t val)
        {
            mTotalNetWt = val;
        }
        size_t getTotalNetWt() const
        {
            return mTotalNetWt;
        }
        void setTotalNeighborWt(size_t val)
        {
            mTotalNeighborWt = val;
        }
        size_t getTotalNeighborWt() const
        {
            return mTotalNeighborWt;
        }
        void setTotNeighbLWt(size_t val)
        {
            mTotNeighbLWt = val;
        }
        size_t getTotNeighbLWt() const
        {
            return mTotNeighbLWt;
        }
        void setTotNeighbBWt(size_t val)
        {
            mTotNeighbBWt = val;
        }
        size_t getTotNeighbBWt() const
        {
            return mTotNeighbBWt;
        }
        void setTotNeighbDWt(size_t val)
        {
            mTotNeighbDWt = val;
        }
        size_t getTotNeighbDWt() const
        {
            return mTotNeighbDWt;
        }

        void setNets(const NetIdVec& nv)
        {
            mNetsIds = nv;
        }
        const NetIdVec& getNetsIds() const
        {
            return mNetsIds;
        }
        void setNeighbors(const CellInstIdVec& cv)
        {
            mNeighbors = cv;
        }
        const CellInstIdVec& getNeighbors() const
        {
            return mNeighbors;
        }
        void setNeighborFixed(bool val)
        {
            mIsNeighborFixed = val;
        }
        bool isNeighborFixed() const
        {
            return mIsNeighborFixed;
        }

        unsigned getNumNets() const
        {
            return mNetsIds.size();
        }
        unsigned getNumNeighbors() const
        {
            return mNeighbors.size();
        }
        void setCurrentPart(unsigned val)
        {
            mCurrPartition = val;
        }
        unsigned getCurrentPart() const
        {
            return mCurrPartition;
        }
        bool isPartValid() const
        {
            return mCurrPartition != UINT_MAX;
        }

    private:
        unsigned        mInstId;
        NetIdVec        mNetsIds; // Ordered by weight
        CellInstIdVec   mNeighbors;
        unsigned        mMasterId; // When cell instances are collapsed to form a larger inst
        CellInstIdVec   mLeaves; // All the leaf level instances contained in this
        unsigned        mLutWt; // Cumulative weight of all leaf cells
        unsigned        mBMemWt; // Cumulative weight of all leaf cells
        unsigned        mDMemWt; // Cumulative weight of all leaf cells
        size_t          mTotalNetWt;
        size_t          mTotalNeighborWt; // Later enumerate to LUT/FLOP/MEM/BB weights
        size_t          mTotNeighbLWt;
        size_t          mTotNeighbBWt;
        size_t          mTotNeighbDWt;
        bool            mIsFixed;
        bool            mIsNeighborFixed;
        unsigned        mCurrPartition;
};

class NetlistGraph
{
    // Data filled in connectivity matrix mConnMtx
    public:
        NetlistGraph(const NetDataVec& nData, const InstDataVec& iData, bool unw); // Later add IO data
        ~NetlistGraph() {}
        // clear before adding
        void addNet(unsigned id)
        {
           Net nt(id);
           mNets.emplace_back(nt);
        }
        void setNet(unsigned id)
        {
            assert(id >= 0 && id < mNets.size());
            const Net& exNet = mNets[id];
            if (exNet.getNetId() == id)
                return;
            Net nt(id);
            mNets[id] = nt;
        }
        void setNetName(unsigned id, const std::string& nm)
        {
            assert(id >= 0 && !nm.empty());
            mNetNames[id] = nm;
        }
        void setInst(unsigned id)
        {
            assert(id >= 0 && id < mInsts.size());
            const NLInst& exInst = mInsts[id];
            if (exInst.getInstId() == id)
                return;
            NLInst inst(id);
            mInsts[id] = inst;
        }
        void setInstName(unsigned id, const std::string& nm)
        {
            assert(id >= 0 && !nm.empty());
            mCellInstNames[id] = nm;
        }
        // clear before adding
        void addInst(unsigned id)
        {
            NLInst inst(id);
            mInsts.emplace_back(inst);
        }

        Net& getNet(unsigned id)
        {
            assert(id >= 0 && id < mNets.size());
            return mNets[id];
        }
        const Net& getConstNet(unsigned id) const
        {
            assert(id >= 0 && id < mNets.size());
            return mNets[id];
        }
        NLInst& getInst(unsigned id)
        {
            assert(id >= 0 && id < mInsts.size());
            return mInsts[id];
        }
        const NLInst& getInst(unsigned id) const
        {
            assert(id >= 0 && id < mInsts.size());
            return mInsts[id];
        }
        unsigned getNumNets() const
        {
            return mNets.size();
        }
        unsigned getNumInsts() const
        {
            return mNumInsts;
        }
        float scaleInstLWt(unsigned wt) const
        {
            if (wt < mMinLutWt)
                return 0;
            return (scaleL * (float)(wt - mMinLutWt)) + 1.0;
        }
        float scaleInstBWt(unsigned wt) const
        {
            if (wt < mMinBMemWt)
                return 0;
            return (scaleB * (float)(wt - mMinBMemWt)) + 1.0;
        }
        float scaleInstDWt(unsigned wt) const
        {
            if (wt < mMinDMemWt)
                return 0;
            return (scaleD * (float)(wt - mMinDMemWt)) + 1.0;
        }
        const NetVec& getNets() const
        {
            return mNets;
        }
        const CellInstVec& getInsts() const
        {
            return mInsts;
        }
        const Matrix2D<unsigned>& getInstMtx() const
        {
            return mGridInsts;
        }

    private:
        unsigned        mNumInsts;
        unsigned        mMinLutWt;
        unsigned        mMaxLutWt;
        unsigned        mMinBMemWt;
        unsigned        mMaxDMemWt;
        unsigned        mMinDMemWt;
        unsigned        mMaxBMemWt;
        float           scaleL;
        float           scaleB;
        float           scaleD;
        NetNameVec      mNetNames; // Id to name
        CellInstNameVec mCellInstNames; // Id to name
        NetVec          mNets;
        CellInstVec     mInsts;
        CellInstVec     mIOs; // IO instances
        Matrix2D<unsigned>          mGridInsts; // Netlist adjacency Matrix 
};

class FpgaCell
{
    public:
        FpgaCell(unsigned id, unsigned origId, const CellData& cd)
        {
            assert(id >= 0);
            mId = id;
            mOrigId = origId;
            mLuts = std::get<0>(cd);
            mBMems = std::get<1>(cd);
            mDMems = std::get<2>(cd);
            mDsps = 0;
            mFlops = 0;
            mNumLinks = 0;
            mLinks.reserve(1u);
            mHasIOInterface = false;
            mPartNum = 0;
            mConnNeighbors.reserve(96);
            mUnused = (mLuts == 0 && mBMems == 0 && mDMems == 0);
            mCellCap = std::make_tuple(mLuts, mBMems, mDMems);
            BBox bbx = std::make_tuple(XY(0.0,0.0), XY(0.0,0.0), XY(0.0,0.0), XY(0.0,0.0));
            mCellBBox = bbx;
        }
        ~FpgaCell() {}

        void setCellId(unsigned id)
        {
            assert(id >= 0);
            mId = id;
            mPartNum = id;
        }
        void setCellOrigId(unsigned id)
        {
            assert(id >= 0);
            mOrigId = id;
        }
        void setNumLinks(unsigned numLinks)
        {
            mNumLinks = numLinks;
        }
        void setCellName(const std::string& nm) {
            assert(!nm.empty());
            mFpgaName = nm;
        }
        void setCellLuts(unsigned numLuts)
        {
            mLuts = numLuts;
        }
        void setCellBMems(unsigned numBMems)
        {
            mBMems = numBMems;
        }
        void setCellDMems(unsigned numDMems)
        {
            mDMems = numDMems;
        }
        void setCellFlops(unsigned numFlops)
        {
            mFlops = numFlops;
        }
        void setCellDsps(unsigned numDsps)
        {
            mDsps = numDsps;
        }
        void setNeighbors(const FpgaCellIdVec& nV)
        {
            mConnNeighbors = nV;
        }
        unsigned getCellId() const
        {
            return mId;
        }
        unsigned getCellOrigId() const
        {
            return mOrigId;
        }
        unsigned getNumLuts() const
        {
            return mLuts;
        }
        unsigned getNumBMems() const
        {
            return mBMems;
        }
        unsigned getNumDMems() const
        {
            return mDMems;
        }
        const FpgaCellIdVec& getNeighbors() const
        {
            return mConnNeighbors;
        }
        unsigned getNumNeighbors() const
        {
            return mConnNeighbors.size();
        }
        bool isUnused() const
        {
            return mUnused;
        }
        const CellData& getCellData() const
        {
            return mCellCap;
        }
        void setBBox(const BBox& bb)
        {
            mCellBBox = bb;
        }
        const BBox& getBBox() const
        {
            return mCellBBox;
        }
        void setCenter(const XY& cntr)
        {
            mFpgaCenter = cntr;
        }
        const XY& getCenter() const
        {
            return mFpgaCenter;
        }
        
    private:
        unsigned        mId;
        unsigned        mOrigId;
        std::string     mFpgaName;
        unsigned        mLuts; // Capacity
        unsigned        mFlops;
        unsigned        mBMems;
        unsigned        mDMems;
        unsigned        mDsps;
        unsigned        mMinL;
        unsigned        mMinB;
        unsigned        mMinD;
        unsigned        mMaxL;
        unsigned        mMaxB;
        unsigned        mMaxD;
        float           scaleL;
        float           scaleB;
        float           scaleD;
        unsigned        mNumLinks; // If this is 0, fpga is inactive
        unsigned        mUnused;
        XY              mFpgaCenter;
        BBox            mCellBBox;
        unsigned        mPartNum; // Equivalent part num
        bool            mHasIOInterface;
        LinkIdVec       mLinks;
        FpgaCellIdVec   mConnNeighbors; // Sorted by distance
        CellData        mCellCap;

};

class FpgaPairLink
{
    public:
        FpgaPairLink(unsigned id)
        {
            mLinkId = id;
            mWt = 0;
            mBw = 0;
            mIsConn = true;
            mFpgaPair = std::make_pair(0,0);
        }
        
        ~FpgaPairLink() {}

        void setWeight(unsigned wt)
        {
            mWt = wt;
        }
        void setBW(unsigned bw)
        {
            mBw = bw;
        }
        void setConn(bool val)
        {
            mIsConn = val;
        }
        void setSrcDestFpgas(unsigned src, unsigned dest)
        {
            assert(src >= 0 && dest >= 0);
            mFpgaPair = std::make_pair(src,dest);
        }

        unsigned getLinkId() const
        {
            return mLinkId;
        }
        unsigned getWeight() const
        {
            return mWt;
        }
        unsigned getBW() const
        {
            return mBw;
        }
        bool isConn() const
        {
            return mIsConn;
        }
        const SourceDestFpgaPair& getFpgaPair() const
        {
            return mFpgaPair;
        }
    private:
        unsigned            mLinkId;
        unsigned            mWt;
        unsigned            mBw;
        bool                mIsConn;
        SourceDestFpgaPair  mFpgaPair;
};

class FpgaSystemDB
{
    // Contains a grid of all connected FPGAs
    // Inactivated FPGAs are marked with a fence
    // Contains weighted links between connected pairs
    //
    public:
        FpgaSystemDB(const LinkDataMap& lDataMap,
                        const CellDataVec& cellDatV,
                        const SysDefData& sysDat);
        ~FpgaSystemDB() {}
        void setFpgasAndLinks(unsigned numFpgas);
        void setupNetlistAndSystemMtx(const NetlistGraph& ng);
        void addFpgaCell(unsigned id, unsigned origId, const CellData& cd)
        {
            assert(id >= 0 && id < mFpgaCells.size());
            assert(origId >= 0);
            const FpgaCell& exCell = mFpgaCells[id];
            if (exCell.getCellId() == id)
                return;
            FpgaCell cell(id, origId, cd);
            mFpgaCells[id] = cell;
            mActive2OrigCellIds[origId] = id;
        }
        void addLink(const FpgaPairLink& fpl)
        {
            mLinks.emplace_back(fpl);
        }

        unsigned getNumFpgas() const
        {
            return mNumFpgas;
        }
        unsigned getDefNumFpgas() const
        {
            return mDefNumFpgas;
        }
        const LinkVec& getLinks() const
        {
            return mLinks;
        }
        const FpgaPairLink& getLink(unsigned id) const
        {
            assert(id < mLinks.size());
            return mLinks[id];
        }
        const FpgaCell& getCell(unsigned id) const
        {
            assert(id >= 0 && id < mFpgaCells.size());
            return mFpgaCells[id];
        }
        FpgaCell& getCell(unsigned id)
        {
            assert(id >= 0 && id < mFpgaCells.size());
            return mFpgaCells[id];
        }
        unsigned getIdFromOrigId(unsigned id) const
        {
            assert (id >= 0 && id < mActive2OrigCellIds.size());
            return mActive2OrigCellIds[id];

        }
        unsigned getLinkId(unsigned src, unsigned dst) const
        {
            return mGridLinks(src, dst);
        }
        const Matrix2D<unsigned>& getSysMtx() const
        {
            return mGridLinks;
        }
        unsigned getCellLCap() const
        {
            return mDCellLCap;
        }
        unsigned getCellBCap() const
        {
            return mDCellBCap;
        }
        unsigned getCellDCap() const
        {
            return mDCellDCap;
        }
        const CellData& getPartCap(unsigned part) const
        {
            const FpgaCell& fc = getCell(part);
            return fc.getCellData();

        }
        bool isCapViol(unsigned part, const CellData& currWt, float thresh) const
        {
            const CellData& partCap = getPartCap(part);
            return (std::get<0>(currWt) > (thresh * std::get<0>(partCap)) ||
                    std::get<1>(currWt) > (thresh * std::get<1>(partCap)) ||
                    std::get<2>(currWt) > (thresh * std::get<2>(partCap)));

        }
        bool isConnectedPair(unsigned i, unsigned j) const
        {
            if (i == j)
                return true;
            if (i == UINT_MAX || j == UINT_MAX)
                return false;
            unsigned lnki = mGridLinks(i,j);
            unsigned lnkj = mGridLinks(j,i);
            if (lnki == UINT_MAX && lnkj == UINT_MAX)
                return false;
            if (lnki != UINT_MAX) {
                const FpgaPairLink& fpli = getLink(lnki);
                if (fpli.isConn() && fpli.getBW() > 0)
                    return true;
            }
            if (lnkj != UINT_MAX) {
                const FpgaPairLink& fplj = getLink(lnkj);
                if (fplj.isConn() && fplj.getBW() > 0)
                    return true;
            }
            return false;
        }
        unsigned getBandwidth(unsigned i, unsigned j) const
        {
            if (i == j)
                return UINT_MAX;
            if (i == UINT_MAX || j == UINT_MAX)
                return 0;
            unsigned lnki = mGridLinks(i,j);
            unsigned lnkj = mGridLinks(j,i);
            if (lnki == UINT_MAX && lnkj == UINT_MAX)
                return 0;
            if (lnki != UINT_MAX) {
                const FpgaPairLink& fpli = getLink(lnki);
                unsigned bwi = fpli.getBW();
                if (fpli.isConn() && bwi > 0)
                    return bwi;
            }
            if (lnkj != UINT_MAX) {
                const FpgaPairLink& fplj = getLink(lnkj);
                unsigned bwj = fplj.getBW();
                if (fplj.isConn() && bwj > 0)
                    return bwj;
            }
            return 0;
        }
        const FpgaCellVec& getCells() const
        {
            return mFpgaCells;
        }
        const Matrix2D<double>& getDistMtx() const
        {
            return mFpgaDist;
        }
        double getWDist(unsigned i, unsigned j) const
        {
            return mFpgaDist(i,j);
        }
    private:
        unsigned                    mNumFpgas;
        unsigned                    mDefNumFpgas;
        unsigned                    mNumFpgasPerBoard;
        unsigned                    mNumBoards;
        unsigned                    mNumRacks;
        unsigned                    mDCellLCap;
        unsigned                    mDCellBCap;
        unsigned                    mDCellDCap;
        std::vector<bool>           mActiveFpgas; // Vector of all FPGAs indicating active/inactive
        LinkVec                     mLinks; 
        FpgaCellVec                 mFpgaCells;
        CellActive2OrigIds          mActive2OrigCellIds;
        Matrix2D<unsigned>          mGridFpgas; // grid of part Ids
        Matrix2D<unsigned>          mGridLinks; // Link Id matrix
        Matrix2D<double>            mFpgaDist; // Weighted distance between 2 fpga centers
};


class AnalyticalPartitioner
{
    public:
        explicit AnalyticalPartitioner(const NetlistGraph& ng, const FpgaSystemDB& fsdb, bool dbg)
                                 : mDbg(dbg), mNg(&ng), mFsdb(&fsdb)
        {
            mNumFpgas = mFsdb->getNumFpgas();
            mNumInsts = mNg->getNumInsts();
            mFpgaNumInsts.reserve(mNumFpgas);
            mFpgaNumInsts.assign(mNumFpgas, 0); // Number of insts in part
            mAllocatedInsts.reserve(mNumInsts);
            mAllocatedInsts.assign(mNumInsts, UINT_MAX); // Mark all Insts unallocated
            mPartWeights.reserve(mNumFpgas);
            mPartWeights.assign(mNumFpgas, std::make_tuple(0,0,0));
            mInstProps.reserve(mNumInsts);
            mInstProps.assign(mNumInsts, std::make_tuple(false, false, false, UINT_MAX));
            mFpgaInsts.reserve(mNumFpgas);
            mFpgaInsts.assign(mNumFpgas, std::unordered_set<unsigned>());
            createSimMatricesAndClusters();
            collectFpgaBridges();
            initialPlacement();
        }
        ~AnalyticalPartitioner() {}
        void createSimMatricesAndClusters();
        void collectFpgaBridges();
        void initialPlacement();

        const FpgaSystemDB* getFsdb() const
        {
            return mFsdb;
        }
        const NetlistGraph* getNg() const
        {
            return mNg;
        }
        unsigned getNumFpgas() const
        {
            return mNumFpgas;
        }
        unsigned getNumInsts() const
        {
            return mNumInsts;
        }
        const MatrixXf& getCoSimFpgas() const
        {
            return mCoSimFpgas;
        }
        const ClusterId2Clust& getClusters() const
        {
            return mClusters;
        }
        const MatrixXf& getCoSimInsts() const
        {
            return mCoSimInsts;
        }
        const MatrixXf& getCoSimInstFpgas() const
        {
            return mCoSimInstFpgas;
        }
        const FpgaBridgeMtx& getFpgaBridges() const
        {
            return mFpgaBridges;
        }
        CellData& getPartWeights(unsigned partId)
        {
            return mPartWeights[partId];
        }
        const CellData& getPartWeights(unsigned partId) const
        {
            return mPartWeights[partId];
        }
        const CellDataVec& getAllPartWts() const
        {
            return mPartWeights;
        }
        void setInstPart(unsigned instId, unsigned partId)
        {
            assert (partId != UINT_MAX && partId < mNumFpgas);
            mAllocatedInsts[instId] = partId;
            mFpgaNumInsts[partId]++;
        }
        void resetInstPart(unsigned instId)
        {
            mAllocatedInsts[instId] = UINT_MAX;
        }
        unsigned getInstPart(unsigned nid) const
        {
            return mAllocatedInsts[nid];
        }
        bool isAllocatedPart(unsigned nid) const
        {
            return mAllocatedInsts[nid] != UINT_MAX;
        }
        const PartNumInstsVec& getPartInsts() const
        {
            return mFpgaNumInsts;
        }
        const InstToPartVec& getInstParts() const
        {
            return mAllocatedInsts;
        }
        void removeInstFromPart(unsigned instId, unsigned partId);
        void addToPartWts(unsigned partId, const CellData& cd);
        void getClusterOutNets(unsigned id, NetIdVec& nidv) const;

        void saveDBasMetisGraph();
        void readHMETISResults();
        bool checkAndAddInstToPart(unsigned instId, unsigned partId, float thresh);
        void updateInstAllocation(unsigned nId, unsigned pId);
        bool canBeAddedToPart(unsigned nId, unsigned pId) const;
        

    private:
        bool        mDbg;
        const       NetlistGraph* mNg;
        const       FpgaSystemDB* mFsdb;
        unsigned    mNumFpgas;
        unsigned    mNumInsts;
        MatrixXf    mCoSimFpgas;
        MatrixXf    mCoSimInsts;
        MatrixXf    mCoSimInstFpgas;
        MatrixXf    mSysEigenVec;
        MatrixXf    mNlEigenVec;
        FpgaProps   mFpgaProps; // isUnused, orig id, centroid
        InstProps   mInstProps; // isCentroid, no-fanin/no-fanout, isFixed, currPart
        ClusterId2Clust  mClusters; // Vector of clusters-vec - ordered by distance to centroid
        ClusterSetVec    mClusterSets; // set of instances for each cluster
        FpgaBridgeMtx    mFpgaBridges;
        ClusterCentroid2Insts   mClusterCentroidInstMap;
        PartNumInstsVec         mFpgaNumInsts; // Number of insts in part
        InstToPartVec   mAllocatedInsts; // Mark all Insts unallocated
        CellDataVec     mPartWeights;
        PartToInstsSV   mFpgaInsts;

};

class JsonReader {
    public:
        explicit JsonReader();
        ~JsonReader() {}
        void readFile(const std::string& f1Name, const std::string& f2Name,
                      LinkDataMap& linkDataMap, CellDataVec& cellDatV,
                      SysDefData& sysDat, InstDataVec& iData, NetDataVec& nData);
        void readFpgaSystem(const std::string& sysFile, LinkDataMap& linkDataMap,
                            CellDataVec& cellDatV, SysDefData& sysDat);
        void readInstanceGraph(const std::string& igFile, InstDataVec& iData, NetDataVec& nData);
    private:
        std::vector<std::string> mFileNames;// board config and netlist graph
};


} // End Namespace SpectralSolver
