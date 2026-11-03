//////////////////////////////////////////////////////////////////////////
// Author:  Anirudh Iyengar
// Date:    July 15, 2025
//
// A spectral K-Way partitioner using multipe eigen vectors
//
// Introduction
// The hierarchical MultiLevel partitioning methods for hypergraphs
// like HMETIS use three phases - COARSENING, INITIAL-PARTITIONING and REFINEMENT
// The INITIAL-PARTITIONING phase for a k-way partitioner generally uses multiple
// random partitioning which is non deterministic. This spectral partitioner
// uses multiple eigen vectors (k eigen vectors for k-way) and clusters them
// with the k-means++ algorithm. Uses the Eigen library.
// To improve on cut refinement, reinforcement learning can be used.
// The current use case has constraints on the partitions. The netlist is
// partitioned and placed into multiple FPGAs connected together on a board.
// The bandwidth between the FPGAs is fixed. The FPGA capacity is fixed and the
// number of LUTs, BMEMs, DMEMs etc are also fixed per FPGA. The FPGAs are homogeneous.
// The number of FPGAs is the number of partitions.
// The FPGA configuration on the board is defined in "blades.json".
// The netlist is in the clustered form of a hypergraph. It is specified in
// "igraph.json" (hypergraph obtained after COARSENING from a multilevel partitioner)
//////////////////////////////////////////////////////////////////////////

#include "spectral_partitioner.h"
#include "rl_refiner.h"

namespace SpectralSolver {

SysPhysAttr resolveSPA(std::string attr) {
    static const std::map<std::string, SysPhysAttr> spAttrStrings {
        { "n", SPA_FPGA },
        { "default_luts_util", SPA_LUTS },
        { "fpga_usable_luts_vec", SPA_ULUTS },
        { "default_bmem_util", SPA_BMEM },
        { "fpga_usable_bmem_vec", SPA_UBMEM },
        { "default_dmem_util", SPA_DMEM },
        { "fpga_usable_dmem_vec", SPA_UDMEM },
        { "fpga_pair_link_data", SPA_LINK_DATA },
        { "bw", SPA_BW },
        { "unconn", SPA_UNCONN },
        { "wt", SPA_LINK_WT }
    };

    const auto itr = spAttrStrings.find(attr);
    if( itr != spAttrStrings.end() ) {
        return itr->second;
    }
    return SPA_MAX; 
}

InstGraphAttr resolveIGA(std::string attr) {
    static const std::map<std::string, InstGraphAttr> igAttrStrings {
        { "nid", IGA_NUM_INSTS },
        { "eid", IGA_NUM_NETS },
        { "num_total_nets", IGA_TOTAL_DNETS },
        { "graph_has_bmem", IGA_HAS_BMEM },
        { "graph_has_dmem", IGA_HAS_DMEM },
        { "node_edge_map", IGA_INST_NET_MAP },
        { "node_names", IGA_INST_NAMES },
        { "all_nodes", IGA_ALL_INSTS },
        { "all_edges", IGA_ALL_NETS },
        { "node_luts_count", IGA_INST_LUT_COUNT },
        { "node_bmem_count", IGA_INST_BMEM_COUNT},
        { "node_dmem_count", IGA_INST_DMEM_COUNT },
        { "edge_nets_count", IGA_EDGE_NETS_COUNT }
    };

    const auto itr = igAttrStrings.find(attr);
    if( itr != igAttrStrings.end() ) {
        return itr->second;
    }
    return IGA_MAX; 
}

static VectorXf::Index
extractMinCoeffIdx(VectorXf& valVec, const std::vector<unsigned>& allocatedInsts)
{
    VectorXf::Index minIdx;
    if (valVec.size() == 0)
        return minIdx;
    float mv = valVec.minCoeff(&minIdx);
    if (allocatedInsts[minIdx] == UINT_MAX)
        return minIdx;
    size_t vsz = valVec.size();
    
    for (int i = 0; i < vsz; i++) {
        if (allocatedInsts[i] != UINT_MAX) {
            valVec[i] = 1.0;
            continue;
        }
    }
    mv = valVec.minCoeff(&minIdx);
    return minIdx;
}

static VectorXf::Index
extractMaxCoeffIdx(VectorXf& valVec, const std::vector<unsigned>& allocatedInsts)
{
    VectorXf::Index maxIdx;
    if (valVec.size() == 0)
        return maxIdx;
    float mv = valVec.maxCoeff(&maxIdx);
    if (allocatedInsts[maxIdx] == UINT_MAX)
        return maxIdx;
    size_t vsz = valVec.size();
    
    for (int i = 0; i < vsz; i++) {
        if (allocatedInsts[i] != UINT_MAX) {
            valVec[i] = -1.0;
            continue;
        }
    }
    mv = valVec.maxCoeff(&maxIdx);
    return maxIdx;
}

static float
findMedian(const std::vector<float>& valVec)
{
    std::vector<float> vals(valVec);
    if (vals.size() < 1)
        return std::numeric_limits<float>::signaling_NaN();
    if (vals.size() == 1)
        return vals[0];

    const auto alpha = vals.begin();
    const auto omega = vals.end();

    // Find the two middle positions (they will be the same if size is odd)
    const auto i1 = alpha + (vals.size()-1) / 2;
    const auto i2 = alpha + vals.size() / 2;

    // Partial sort to place the correct elements at those indexes (it's okay to modify the vector,
    // as we've been given a copy; otherwise, we could use std::partial_sort_copy to populate a
    // temporary vector).
    std::nth_element(alpha, i1, omega);
    std::nth_element(i1, i2, omega);

    return 0.5 * (*i1 + *i2);
}

// NetlistGraph object
NetlistGraph::NetlistGraph(const NetDataVec& nData, const InstDataVec& iData, bool unw)
{
    // Can be formed only after adding all nets and instances
    unsigned maxSz = UINT_MAX;
    Net dummyNet(maxSz);
    NLInst dummyInst(maxSz);

    bool unitNetWeight = unw;
    
    unsigned numNets = nData.size();
    unsigned numInsts = iData.size();
    unsigned numIOs = 1;
    assert(numNets > 0 && numInsts > 0);
    mNetNames.reserve(numNets+1u);
    mNetNames.assign(numNets, "dummy");
    mNets.reserve(numNets+1u);
    mNets.assign(numNets, dummyNet);

    mCellInstNames.reserve(numInsts+1u);
    mCellInstNames.assign(numInsts, "dummy");
    mInsts.reserve(numInsts+1u);
    mInsts.assign(numInsts, dummyInst);

    unsigned nId = 0;
    for (auto& ndt : nData) {
        const NetData& el = ndt;
        mNetNames[nId] = std::get<0>(el);
        setNet(nId);
        unsigned sInstId = std::get<1>(el).first;
        unsigned dInstId = std::get<1>(el).second;
        setInst(sInstId);
        setInst(dInstId);
        Net& nt = getNet(nId);
        nt.setNetWeight(unitNetWeight ? 1 : std::get<2>(el));
        //std::cout << "SET NW: " << nId << " - " << std::get<2>(el) << " -- " << sInstId << "/" << dInstId << std::endl;
        nt.setSrcDestInstId(sInstId, dInstId); 
        nId++;
    }

    unsigned instId = 0;
    unsigned mnL = UINT_MAX, mxL = 0,
             mnB = UINT_MAX, mxB = 0,
             mnD = UINT_MAX, mxD = 0;
    for (auto& idt : iData) {
        const InstData& el = idt;
        mCellInstNames[instId] = std::get<0>(el);
        setInst(instId);
        NLInst& ci = getInst(instId);
        const NetIdVec& nv = std::get<1>(el);
        ci.setNets(nv);
        unsigned lw = std::get<2>(el);
        ci.setLutWt(lw);
        if (lw < mnL && lw > 0)
            mnL = lw;
        if (lw > mxL && lw > 0)
            mxL = lw;
        unsigned bw = std::get<3>(el);
        ci.setBMemWt(bw);
        if (bw < mnB && bw > 0)
            mnB = bw;
        if (bw > mxB && bw > 0)
            mxB = bw;
        unsigned dw = std::get<4>(el);
        ci.setDMemWt(dw);
        if (dw < mnD && dw > 0)
            mnD = dw;
        if (dw > mxD && dw > 0)
            mxD = dw;
        //std::cout << "LW/BW/DW: " << lw << "/" << bw << "/" << dw << std::endl;
        instId++;
    }

    mMinLutWt = mnL == UINT_MAX ? 0 : mnL;
    mMaxLutWt = mxL == 0 ? 1 : mxL;
    mMinBMemWt = mnB == UINT_MAX ? 0 : mnB;
    mMaxBMemWt = mxB == 0 ? 1 : mxB;
    mMinDMemWt = mnD == UINT_MAX ? 0 : mnD;
    mMaxDMemWt = mxD == 0 ? 1 : mxD;

    unsigned lDiff = mMaxLutWt - mMinLutWt;
    scaleL = lDiff > 0 ? (99.0 /(float)(lDiff)) : 0.001;
    unsigned bDiff = mMaxBMemWt - mMinBMemWt;
    scaleB = bDiff > 0 ? (99.0 /(float)(bDiff)) : 0.001;
    unsigned dDiff = mMaxDMemWt - mMinDMemWt;
    scaleD = dDiff > 0 ? (99.0 /(float)(dDiff)) : 0.001;
    //std::cout << "L/B/D" << mMinLutWt << "/" << mMaxLutWt << "=" << mMinBMemWt << "/" << mMaxBMemWt << "=" << mMinDMemWt << "/" << mMaxDMemWt << std::endl;
    //std::cout << "SCALES: " << scaleL << "/" << scaleB << "/" << scaleD << std::endl;
    for (unsigned instIdx = 0; instIdx < numInsts; instIdx++) {
        NLInst& ci = getInst(instIdx);
        const NetIdVec& nv = ci.getNetsIds();
        CellInstIdVec neighbors;
        neighbors.reserve(nv.size());
        size_t totalNtWt = 0;
        size_t totalNeighborWt = 0;
        size_t totNeighbLWt = 0;
        size_t totNeighbBWt = 0;
        size_t totNeighbDWt = 0;
        for (auto& nId : nv) {
            const Net& nt = getNet(nId);
            unsigned sId = nt.getNetSrc();
            unsigned dId = nt.getNetDest();
            unsigned nghId = sId == instIdx ? dId : sId;
            const NLInst& nci = getInst(nghId);
            totalNeighborWt += nci.getTotalWeight();
            totNeighbLWt += nci.getLutWeight();
            totNeighbBWt += nci.getBMemWeight();
            totNeighbDWt += nci.getDMemWeight();
            totalNtWt += nt.getWeight();;
            neighbors.emplace_back(nghId);
        }
        ci.setNeighbors(neighbors);
        ci.setTotalNeighborWt(totalNeighborWt);
        ci.setTotNeighbLWt(totNeighbLWt);
        ci.setTotNeighbBWt(totNeighbBWt);
        ci.setTotNeighbDWt(totNeighbDWt);
        ci.setTotalNetWt(totalNtWt);
    }

    mIOs.reserve(numIOs+1u);
    if (numIOs > 0) {
        mIOs.assign(numIOs, dummyInst);
    }

    mNumInsts = mInsts.size();;
    mGridInsts = Matrix2D<unsigned>(mNumInsts, mNumInsts); // Adjacency matrix
    mGridInsts.fill(UINT_MAX);
    for (unsigned i = 0; i < numNets; i++) {
        const Net& nt = getConstNet(i);
        const SourceDestInstPair& sdP = nt.getNetSrcDest();
        mGridInsts(sdP.first, sdP.second, nt.getNetId());
        mGridInsts(sdP.second, sdP.first, nt.getNetId());
    }
    for (unsigned j = 0; j < numInsts; j++) {
        for (unsigned k = 0; k < numInsts; k++) {
            if (j == k)
                assert (mGridInsts(j,k) == UINT_MAX);
        }
    }
}

FpgaSystemDB::FpgaSystemDB(const LinkDataMap& lDataMap, const CellDataVec& cellDatV,
                                 const SysDefData& sysDat)
{
    assert(lDataMap.size() > 1 && lDataMap.size() == cellDatV.size());
    unsigned defNumFpgas = std::get<0>(sysDat);
    mDefNumFpgas = defNumFpgas > 0 ? defNumFpgas : lDataMap.size();

    assert (mDefNumFpgas > 2 && mDefNumFpgas < 3000);
    mActiveFpgas.reserve(mDefNumFpgas);
    mActiveFpgas.assign(mDefNumFpgas, true);

    mNumFpgas = mDefNumFpgas - std::get<4>(sysDat);
    if (mNumFpgas < 2 || mNumFpgas > 3000) {
        std::cout << "ERROR: Number of FPGAs cannot be less than 2 or greater than 3000" << std::endl;
        assert(mNumFpgas > 1 && mNumFpgas <= 3000);
    }
    // Initialize with defaults
    mNumFpgasPerBoard = 6;
    mNumBoards = std::ceil((double)mNumFpgas / mNumFpgasPerBoard);

    mDCellLCap = std::get<1>(sysDat);
    assert(mDCellLCap > 0);

    mDCellBCap = std::get<2>(sysDat);
    mDCellDCap = std::get<3>(sysDat);

    unsigned numRacks = std::ceil((double)mNumBoards / 8);
    mNumRacks = numRacks;
    unsigned numFpgaCols = numRacks * 3;
    assert(numFpgaCols > 0);
    unsigned numFpgaRows = 16;
    assert(numFpgaRows > 0);
    mActiveFpgas = std::vector<bool>(mNumFpgas, true);
    mFpgaCells.reserve(mNumFpgas);
    CellData cd = std::make_tuple(0,0,0);
    mFpgaCells.assign(mNumFpgas, FpgaCell(UINT_MAX, UINT_MAX, cd));

    mActive2OrigCellIds.reserve(mDefNumFpgas);
    mActive2OrigCellIds.assign(mDefNumFpgas, UINT_MAX);

    mGridFpgas = Matrix2D<unsigned>(numFpgaRows, numFpgaCols); // Fpga inst matrix
    mGridFpgas.fill(UINT_MAX);

    mFpgaDist = Matrix2D<double>(mNumFpgas, mNumFpgas);
    mFpgaDist.fill(DBL_MAX); //Grid cell width/height is 2.0x2.0
    
    mGridLinks = Matrix2D<unsigned>(mNumFpgas, mNumFpgas); // Link Id matrix
    mGridLinks.fill(UINT_MAX);
    mLinks.reserve(mNumFpgas*mNumFpgas + 1u);

    std::unordered_set<unsigned> unusedFpgas;
    int fcId = 0, origId = -1;
    for (auto& cl : cellDatV) {
        origId++;
        if (std::get<0>(cl) == 0 &&
            std::get<1>(cl) == 0 &&
            std::get<2>(cl) == 0) {
            unusedFpgas.insert(origId);
            continue;
        }
        addFpgaCell(fcId, origId, cl);
        fcId++;
    }

    unsigned linkId = 0;
    for (auto& lDat : lDataMap) {
        unsigned id = lDat.first;
        if (unusedFpgas.find(id) != unusedFpgas.end())
            continue;
        unsigned aSrcId = getIdFromOrigId(id);
        unsigned numLinks = 0;
        FpgaCellIdVec neighborVec;
        const std::vector<LinkData>& ldv = lDat.second;
        for (auto& el : ldv) {
            const LinkData& ldata = el;
            unsigned destId = std::get<0>(ldata);
            if (unusedFpgas.find(destId) != unusedFpgas.end())
                continue;
            unsigned aDestId = getIdFromOrigId(destId);
            FpgaPairLink fpl(linkId);
            unsigned bw = std::get<1>(ldata);
            unsigned wt = std::get<2>(ldata);
            bool unconn = std::get<3>(ldata);
            fpl.setSrcDestFpgas(aSrcId, aDestId);
            fpl.setBW(bw);
            fpl.setWeight(wt);
            fpl.setConn(!unconn);
            addLink(fpl);
            linkId++;
            numLinks++;
            neighborVec.emplace_back(aDestId);
        }
        unsigned aId = getIdFromOrigId(id);
        assert(aId != UINT_MAX);
        FpgaCell& fcl = getCell(aId);
        fcl.setNumLinks(numLinks);
        fcl.setNeighbors(neighborVec);
    }

    for (auto& link : mLinks) {
        const FpgaPairLink& fpl = link;
        const SourceDestFpgaPair& sdPair = fpl.getFpgaPair();
        //std::cout << "Adding link: " << fpl.getLinkId() << " -- " << sdPair.first << "/" << sdPair.second << std::endl;
        mGridLinks(sdPair.first, sdPair.second, fpl.getLinkId());
    }

    // Setup physical graph with coordinates
    for (unsigned i = 0; i < numFpgaRows; i++) {
        for (unsigned j = 0; j < numFpgaCols; j++) {
            unsigned k = j/3; // Rack number
            unsigned part = i*3 + (j + (k*45));
            mGridFpgas(i,j,part);
            if (part >= mNumFpgas)
                continue;
            FpgaCell& cl = getCell(part);
            XY ll((j*2.0), (i*2.0));
            XY lr(ll.X()+2.0, ll.Y());
            XY ul(ll.X(), ll.Y()+2.0);
            XY ur(ll.X()+2.0, ll.Y()+2.0);
            XY cntr(ll.X()+1.0, ll.Y()+1.0);
#if 0
            std::cout << "Setting for part: " << part <<  " -- "
                      << ll.X() << "," << ll.Y() << "/"
                      << lr.X() << "," << lr.Y() << "/"
                      << ul.X() << "," << ul.Y() << "/"
                      << ur.X() << "," << ur.Y() << "/"
                      << cntr.X() << "," << cntr.Y()
                      << std::endl;
#endif
            BBox bbx = std::make_tuple(ll,lr,ul,ur);
            cl.setBBox(bbx);
            cl.setCenter(cntr);
        }
    }

    for (unsigned i = 0; i < mNumFpgas; i++) {
        for (unsigned j = 0; j < mNumFpgas; j++) {
            if (i == j) {
                mFpgaDist(i,j, 0.0);
                continue;
            }
            const FpgaCell& cl1 = getCell(i);
            const FpgaCell& cl2 = getCell(j);
            const XY& ctr1 = cl1.getCenter();
            const XY& ctr2 = cl2.getCenter();
            double edist = ctr1.getEDist(ctr2);
            assert(edist > 0.0);
            // Weighted dist is the BW*edist
            double wdist = DBL_MAX;
            unsigned bw = 0;
            if (isConnectedPair(i,j)) {
                bw = getBandwidth(i,j); // SHOULD be scaling factor times edist
                wdist = (bw == UINT_MAX || bw == 0) ? 10e6 : (((double)bw) * edist);
            } else {
                wdist = 10e6;
            }
            //std::cout << "I/j: " << i << "," << j << " -- " << ctr1.X() << "/" << ctr1.Y() << "," << ctr2.X() << "/" << ctr2.Y() << std::endl;
            mFpgaDist(i,j,wdist);
            //std::cout << "FPGA DIST: " << i << "," << j << " -- " << wdist << " -- " << edist << "/" << bw << std::endl;
        }
    }
}

/////////////////// READ JSON FILES ///////////////////////
JsonReader::JsonReader ()
{
    mFileNames.reserve(2);
}

void JsonReader::readFpgaSystem(const std::string& sysFile, LinkDataMap& linkDataMap, CellDataVec& cellDataV, SysDefData& sysDat)
{
    if (sysFile.empty() || sysFile == "")
        return;
    std::ifstream jfile(sysFile);
    json jp = json::parse(jfile);

    unsigned numFpgas = 0;
    unsigned lutCap = 0,
             bMemCap = 0,
             dMemCap = 0;
    std::vector<unsigned> lutV;
    std::vector<unsigned> bmV;
    std::vector<unsigned> dmV;

    for (const auto& it : jp.items()) {
        // the value
        json &val = it.value();
    
        // the key (for objects)
        const std::string &key = it.key();

        std::cout << key << " ---- : ----" << std::endl;
        for (const auto& el : val.items()) {
            json& v1 = el.value();
            const std::string& k1 = el.key();
            switch (resolveSPA(k1))
            {
                case SPA_FPGA:
                    {
                        std::cout << "Number of FPGAs: " << v1.dump(4) << std::endl;
                        numFpgas = v1.get<unsigned>();
                    }
                    break;
                case SPA_LUTS:
                    {
                        std::cout << "Number of default LUTs: " << v1.dump(4) << std::endl;
                        lutCap = v1.get<unsigned>();
                    }
                    break;
                case SPA_ULUTS:
                    {
                        for (const auto& lel : v1.items()) {
                            json& lv = lel.value();
                            //std::cout << lv.dump(4) << std::endl;
                            unsigned luts = lv.get<unsigned>();
                            lutV.emplace_back(luts);
                        }
                    }
                    break;
                case SPA_BMEM:
                    {
                        std::cout << "Number of default BMEMs: " << v1.dump(4) << std::endl;
                        bMemCap = v1.get<unsigned>();
                    }
                    break;
                case SPA_UBMEM:
                    {
                        for (const auto& bel : v1.items()) {
                            json& bv = bel.value();
                            //std::cout << bv.dump(4) << std::endl;
                            unsigned bmems = bv.get<unsigned>();
                            bmV.emplace_back(bmems);
                        }
                    }
                    break;
                case SPA_DMEM:
                    {
                        std::cout << "Number of default DMEMs: " << v1.dump(4) << std::endl;
                        dMemCap = v1.get<unsigned>();
                    }
                        break;
                case SPA_UDMEM:
                    {
                        for (const auto& del : v1.items()) {
                            json& dv = del.value();
                            //std::cout << dv.dump(4) << std::endl;
                            unsigned dmems = dv.get<unsigned>();
                            dmV.emplace_back(dmems);
                        }
                    }
                    break;
                case SPA_LINK_DATA:
                    {
                        int i = 0, j = 0;
                        for (const auto& el1 : v1.items()) {
                            json& v2 = el1.value();
                            // Each pair 0 to (n*n)-1
                            std::vector<LinkData> ldV;
                            for (const auto& el2 : v2.items()) {
                                json& v3 = el2.value();
                                //std::cout << "Key for link data: " << i << "," << j << std::endl;
                                LinkData ld = std::make_tuple(j,0,0,false);
                                for (const auto& el3 : v3.items()) {
                                    const std::string& lpa = el3.key();
                                    switch (resolveSPA(lpa))
                                    {
                                        case SPA_BW:
                                            {
                                                //std::cout << lpa << " - " << el3.value().dump(4) << std::endl;
                                                std::get<1>(ld) = el3.value().get<unsigned>();
                                            }
                                            break;
                                        case SPA_UNCONN:
                                            {
                                                //std::cout << lpa << " - " << el3.value().dump(4) << std::endl;
                                                std::get<3>(ld) = el3.value().get<bool>();
                                            }
                                            break;
                                        case SPA_LINK_WT:
                                            {
                                                //std::cout << lpa << " - " << el3.value().dump(4) << std::endl;
                                                std::get<2>(ld) = el3.value().get<unsigned>();
                                            }
                                            break;
                                            
                                    }
                                }
                                ldV.emplace_back(ld);
                                j++;
                            }
                            linkDataMap.insert(std::make_pair(i, ldV));
                            i++; j = 0;
                        }
                        // Link data
                        
                        for (auto& dt : linkDataMap) {
                            //std::cout << "FPGA: " << dt.first << " - ";
                            const std::vector<LinkData>& lnkDV = dt.second;
                            for (auto& dt1 : lnkDV) {
                                const LinkData& lnkD = dt1;
                                //std::cout << "\t" << std::get<0>(lnkD) << "/" << std::get<1>(lnkD)
                                //<< "/" << std::get<2>(lnkD) << "/" << std::get<3>(lnkD) << std::endl;
                            }
                        }
                    }
                    break;
                default:
                    break;

            }
        }
    }
  
    // Get actually used FPGAs
    std::get<0>(sysDat) = numFpgas;
    std::get<1>(sysDat) = lutCap;
    std::get<2>(sysDat) = bMemCap;
    std::get<3>(sysDat) = dMemCap;

    cellDataV.reserve(numFpgas+1u);
    cellDataV.assign(numFpgas, std::make_tuple(0,0,0));
    int i = 0;

    for (auto& el : lutV) {
        std::get<0>(cellDataV[i]) = el;
        i++;
    }
    i = 0;
    for (auto& el : bmV) {
        std::get<1>(cellDataV[i]) = el;
        i++;
    }
    i = 0;
    for (auto& el : dmV) {
        std::get<2>(cellDataV[i]) = el;
        i++;
    }

    unsigned unUsedFpgas = 0;
    for (auto& el : cellDataV) {
        if (std::get<0>(el) == 0 &&
            std::get<1>(el) == 0 &&
            std::get<2>(el) == 0)
            unUsedFpgas++;
    }
    std::get<4>(sysDat) = unUsedFpgas;
}

void JsonReader::readInstanceGraph(const std::string& igFile, InstDataVec& iData, NetDataVec& nData)
{
    if (igFile.empty() || igFile == "")
        return;
    std::ifstream jfile(igFile);
    json jp = json::parse(jfile);
    unsigned numNets = 0,
           numInsts = 0;
    for (const auto& it : jp.items()) {
        json &val = it.value();
    
        const std::string &key = it.key();
        //std::cout << key << " ---- : ----" << std::endl;
        for (const auto& el : val.items()) {
            json& v1 = el.value();
            const std::string& k1 = el.key();
            if (numNets > 0 && numInsts > 0)
                break;
            switch (resolveIGA(k1))
            {
                case IGA_NUM_INSTS:
                    {
                        //std::cout << "Number of Insts: " << v1.dump(4) << std::endl;
                        numInsts = v1.get<unsigned>();
                    }
                    break;
                case IGA_NUM_NETS:
                    {
                        //std::cout << "Number of nets: " << v1.dump(4) << std::endl;
                        numNets = v1.get<unsigned>();
                    }
                    break;
                default:
                    break;
            }
        }
    }

    // Create netlist graph
    
    // Setup data structures for collecting netlist info
    ///std::vector<InstData> iData; //vector indices are inst ids
    iData.reserve(numInsts+1u);
    NetIdVec ntv;
    InstData dummyI = std::make_tuple("NoName", ntv, 0, 0, 0);
    iData.assign(numInsts, dummyI);

    ///std::vector<NetData> nData; //vector indices are net ids
    nData.reserve(numNets+1u);
    NetData dummyN = std::make_tuple("NoName", std::make_pair(0,0), 0);
    nData.assign(numNets, dummyN);
    

    for (const auto& it : jp.items()) {
        json &val = it.value();
    
        const std::string &key = it.key();
        std::cout << key << " ---- : ----" << std::endl;

        for (const auto& el : val.items()) {
            json& v1 = el.value();
            const std::string& k1 = el.key();
            switch (resolveIGA(k1))
            {
                case IGA_NUM_INSTS:
                    {
                        std::cout << "Number of Insts: " << v1.dump(4) << std::endl;
                    }
                    break;
                case IGA_NUM_NETS:
                    {
                        std::cout << "Number of nets: " << v1.dump(4) << std::endl;
                    }
                    break;
                case IGA_TOTAL_DNETS:
                    {
                        std::cout << "Total nets weight: " << v1.dump(4) << std::endl;
                    }
                    break;
                case IGA_HAS_BMEM:
                    {
                        std::cout << "Graph has BMEMs: " << v1.dump(4) << std::endl;
                    }
                    break;
                case IGA_HAS_DMEM:
                    {
                        std::cout << "Graph has DMEMs: " << v1.dump(4) << std::endl;
                    }
                    break;
                case IGA_INST_NET_MAP:
                    {
                        for (const auto& el1 : v1.items()) { // Inst pair id and net id
                            json& v2 = el1.value(); // inst-pair key and net key
                            for (const auto& el2 : v2.items()) {
                                json& v3 = el2.value();
                                const std::string& ipk = el2.key();
                                if (ipk == "key") {
                                    for (const auto& el3 : v3.items()) {
                                        json& v4 = el3.value();
                                        const std::string& sdk = el3.key();
                                        if (sdk == "first") {
                                            //std::cout << "Insts: " << el3.value().dump(4) << " -- ";
                                        } else if (sdk == "second") {
                                            //std::cout << el3.value().dump(4) << " -- NET: ";
                                        }
                                    }
                                } else if (ipk == "value") {
                                    //std::cout << el2.value().dump(4) << std::endl;
                                }
                            }
                        }
                    }
                    break;
                case IGA_INST_NAMES:
                    break; // Later fill names
                case IGA_ALL_INSTS:
                    {
                        std::vector<unsigned> netVec;
                        netVec.reserve(1024);
                        for (const auto& el1 : v1.items()) {
                            json& v4 = el1.value();
                            unsigned totalNets = 0;
                            netVec.clear();
                            unsigned inst = 0;
                            unsigned lwt = 0,
                                     bwt = 0,
                                     dwt = 0;
                            for (const auto& el4 : v4.items()) {
                                const std::string& iattr = el4.key();
                                if (iattr == "id") {
                                    //std::cout << "INST: " << el4.value().dump(4) << "-";
                                    inst = (el4.value().get<unsigned>());
                                } else if (iattr == "nin") {
                                    //std::cout << "inps: " << el4.value().dump(4) << "-";
                                    totalNets += el4.value().get<unsigned>();
                                } else if (iattr == "nout") {
                                    //std::cout << "outs: " << el4.value().dump(4) << std::endl;
                                    totalNets += el4.value().get<unsigned>();
                                }  else if (iattr == "luts") {
                                    lwt = el4.value().get<unsigned>();
                                } else if (iattr == "bmems") {
                                    bwt = el4.value().get<unsigned>();
                                } else if (iattr == "dmems") {
                                    dwt = el4.value().get<unsigned>();
                                } else if (iattr == "valid_locs") {
                                    // Use it for fixed loc
                                } else if (iattr == "fanins") {
                                    json& val_fi = el4.value();
                                    for (const auto& fi : val_fi.items()) {
                                        netVec.emplace_back(fi.value().get<unsigned>()); 
                                    }
                                } else if (iattr == "fanouts") {
                                    json& val_fo = el4.value();
                                    for (const auto& fo : val_fo.items()) {
                                        netVec.emplace_back(fo.value().get<unsigned>()); 
                                    }
                                }

                            }
                            assert(netVec.size() == totalNets);
                            assert(inst < iData.size());
                            InstData& idat = iData[inst];
                            std::get<0>(idat) = "I" + std::to_string(inst);
                            std::get<1>(idat) = netVec;
                            std::get<2>(idat) = lwt;
                            std::get<3>(idat) = bwt;
                            std::get<4>(idat) = dwt;
                        }
                    }
                    break;
                case IGA_ALL_NETS:
                    {
                        for (const auto& el1 : v1.items()) {
                            json& v5 = el1.value();
                            std::array<unsigned,5> arr = {0,0,0,0,0};
                            for (const auto& el5 : v5.items()) {
                                const std::string& nattr = el5.key();
                                if (nattr == "id") {
                                    //std::cout << "NET: " << el5.value().dump(4) << "-";
                                    arr[0] = el5.value().get<unsigned>();
                                } else if (nattr == "src") {
                                   // std::cout << "src: " << el5.value().dump(4) << "-";
                                    arr[1] = el5.value().get<unsigned>();
                                } else if (nattr == "dst") {
                                    //std::cout << "dst: " << el5.value().dump(4) << "-";
                                    arr[2] = el5.value().get<unsigned>();
                                } else if (nattr == "wt") {
                                    //std::cout << "wt: " << el5.value().dump(4) << std::endl;
                                    arr[3] = el5.value().get<unsigned>();
                                } else if (nattr == "nets") {
                                    //std::cout << "nets: " << el5.value().dump(4) << std::endl;
                                    arr[4] = el5.value().get<unsigned>();
                                }
                            }
                            assert(arr[0] < nData.size());
                            NetData& ndt = nData[arr[0]];
                            std::get<0>(ndt) = "N" + std::to_string(arr[0]);
                            std::get<1>(ndt) = std::make_pair(arr[1], arr[2]);
                            std::get<2>(ndt) = arr[4];
                        }
                    }
                    break;
                case IGA_INST_LUT_COUNT:
                    {
                        unsigned m = 0;
                        for (const auto& el1 : v1.items()) {
                            //std::cout << "INST-" << m << "-LC: " << el1.value().dump(4) << std::endl;
                            unsigned wt = el1.value().get<unsigned>();
                            assert(m < iData.size());
                            InstData& idat = iData[m];
                            std::get<2>(idat) = wt;
                            ++m;
                        }
                    }
                    break;
                case IGA_INST_BMEM_COUNT:
                    {
                        unsigned n = 0;
                        for (const auto& el1 : v1.items()) {
                            //std::cout << "INST-" << n << "-BC: " << el1.value().dump(4) << std::endl;
                            unsigned wt = el1.value().get<unsigned>();
                            assert(n < iData.size());
                            InstData& idat = iData[n];
                            std::get<3>(idat) = wt;
                            ++n;
                        }
                    }
                    break;
                case IGA_INST_DMEM_COUNT:
                    {
                        unsigned p = 0;
                        for (const auto& el1 : v1.items()) {
                            //std::cout << "INST-" << p << "-DC: " << el1.value().dump(4) << std::endl;
                            unsigned wt = el1.value().get<unsigned>();
                            assert(p < iData.size());
                            InstData& idat = iData[p];
                            std::get<4>(idat) = wt;
                            ++p;
                        }
                    }
                    break;
                case IGA_EDGE_NETS_COUNT:
                    {
                        int q = 0;
                        for (const auto& el1 : v1.items()) {
                            //std::cout << "NET-" << q << "-ENC: " << el1.value().dump(4) << std::endl;
                            //std::cout << "NET-" << q << "-ENC: " << el1.value().get<int>() << std::endl;
                            assert(q < nData.size());
                            NetData& ndt = nData[q];
                            int enc = el1.value().get<int>();
                            std::get<2>(ndt) = enc;
                            ++q;
                        }
                    }
                    break;
                default:
                    break;

            }
        }
    }
#if 0
    //iData and nData
    for (auto& idt : iData) {
        const InstData& el = idt;
        std::cout << "Inst: " << std::get<0>(el) << std::endl;
        const NetIdVec& nv = std::get<1>(el);
        for (auto& nt : nv) {
            std::cout << "\t";
            std::cout << "NET: " << nt << std::endl;
        }
        std::cout << "\t" << "LUTs: " << std::get<2>(el) << "/BMEMs: " << std::get<3>(el) << "/DMEMs: " << std::get<4>(el) << std::endl;
    }

    for (auto ndt : nData) {
        const NetData& el = ndt;
        std::cout << "Net: " << std::get<0>(el) << "/" << std::get<1>(el).first << "--" << std::get<1>(el).second << "/" << std::get<2>(el) << std::endl;
    }
#endif
}

void JsonReader::readFile(const std::string& f1Name, const std::string& f2Name,
                          LinkDataMap& linkDataMap, CellDataVec& cellDatV,
                          SysDefData& sysDat, InstDataVec& iData, NetDataVec& nData)
{
    // Check if file exists
    if (f1Name.empty() || f1Name == "") {
        std::cout << "ERROR: File: " << f1Name << " does not exist" << std::endl;
        assert(0);
    }
    if (f2Name.empty() || f2Name == "") {
        std::cout << "ERROR: File: " << f2Name << " does not exist" << std::endl;
        assert(0);
    }
    for (int fIdx = 0; fIdx < 2; fIdx++) {
        std::string fName = fIdx == 0 ? f1Name : f2Name;
        struct stat buff;
        const char* fPath = fName.c_str();
        if (stat(fPath, &buff) < 0) {
            std::cout << "ERROR: File path: " << fName << " does not exist" << std::endl;
            assert(0);
        }

        std::ifstream jFile(fPath);
        if (!jFile.good()) {
            std::cout << "ERROR: File: " << fName << " is corrupt or unreadable" << std::endl;
            jFile.close();
            assert(0);
        }
        unsigned pos = fName.rfind('/', fName.length());
        std::string onlyFile;
        if (pos != std::string::npos) {
            onlyFile = (fName.substr(pos+1, fName.length() - pos));
        }
        if (!onlyFile.empty()) {
            mFileNames.emplace_back(fName);
            if (onlyFile == "blades.json") {
                readFpgaSystem(fName, linkDataMap, cellDatV, sysDat);
            } else if (onlyFile == "igraph.json") {
                readInstanceGraph(fName, iData, nData);
            } else {
                std::cout << "Valid file names are: blades.json or igraph.json" << std::endl;
                assert(0);
            }
        }
    }
}

void AnalyticalPartitioner::createSimMatricesAndClusters()
{
    // Create cosine similarity matrices for insts, fpgas and insts-fpgas 
    std::cout << "Actual number of FPGAs: " << mNumFpgas << std::endl;
    const Matrix2D<unsigned>& sysMtx = mFsdb->getSysMtx();

    std::unordered_set<unsigned> uFpgaSet;
    // Original Integer connection Mtx
    MatrixXi sysConnIMtx(mNumFpgas, mNumFpgas);
    sysConnIMtx = MatrixXi::Zero(mNumFpgas, mNumFpgas);

    // Scaled connection Mtx
    MatrixXf sysConnMtx(mNumFpgas, mNumFpgas);
    sysConnMtx = MatrixXf::Zero(mNumFpgas, mNumFpgas);
   
    // System degree Mtx
    MatrixXf sysDegMtx(mNumFpgas, mNumFpgas);
    sysDegMtx = MatrixXf::Zero(mNumFpgas, mNumFpgas);
    
    MatrixXf rawCellWtMtx(mNumFpgas, mNumFpgas);
    rawCellWtMtx = MatrixXf::Zero(mNumFpgas, mNumFpgas);
    
    MatrixXf cellWtMtx(mNumFpgas, mNumFpgas);
    cellWtMtx = MatrixXf::Zero(mNumFpgas, mNumFpgas);

    // CREATE FPGA SYTSTEM COSIM MTX

    unsigned minLnkWt = UINT_MAX,
             maxLnkWt = 0;
    for (unsigned i = 0; i < mNumFpgas; i++) {
        unsigned rowWt = 0;
        for (unsigned j = 0; j < mNumFpgas; j++) {
            unsigned lId = sysMtx(i,j);
            unsigned otherId = sysMtx(j,i);
            unsigned wt = 0;
            if (lId != UINT_MAX) {
                const FpgaPairLink& lnk = mFsdb->getLink(lId);
                if (lnk.isConn() && lnk.getBW() > 0) {
                    wt = lnk.getBW();
                }
            }
            if (otherId != UINT_MAX) {
                const FpgaPairLink& lnk = mFsdb->getLink(otherId);
                if (lnk.isConn() && lnk.getBW() > 0) {
                    wt = std::max(wt, lnk.getBW());
                }
            }
            if (i == j)
                assert(wt == 0);
            if (wt > 0  && wt > maxLnkWt)
                maxLnkWt = wt;
            if (wt > 0 && wt < minLnkWt)
                minLnkWt = wt;
            sysConnIMtx(i,j) = wt;
            rowWt += wt;
        }
        if (rowWt == 0)
            uFpgaSet.insert(i);
        const FpgaCell& fc = mFsdb->getCell(i);
        unsigned cellWt = fc.getNumLuts() + fc.getNumBMems() + fc.getNumDMems();
        if (cellWt == 0) {
            std::cout << "INFO: " << "FPGA: " << i << " is unused" << std::endl;
            cellWt = 1;
        }
        rawCellWtMtx(i,i) = cellWt;
        cellWtMtx(i,i) = 1 + std::log(cellWt);
    }
    assert(minLnkWt != UINT_MAX && maxLnkWt != 0);

    float lnkScale = (maxLnkWt > minLnkWt) ? (99.0f / (float)(maxLnkWt - minLnkWt)) : 1.0f;
    std::cout << "MIN/MAX LNK WT: " << minLnkWt << "/" << maxLnkWt << " -- " << lnkScale << std::endl;

    for (auto& el : uFpgaSet) {
        std::cout << "INACTIVE FPGAs: " << el << std::endl;
    }
    
    for (unsigned i = 0; i < mNumFpgas; i++) {
        unsigned rowWt = 0;
        for (int j = 0; j < mNumFpgas; j++) {
            if (sysConnIMtx(i,j) != sysConnIMtx(j,i)) {
                std::cout << "SYSCONN NOT SYMM: " << i << "," << j << "--" << sysConnIMtx(i,j) << "/" << sysConnIMtx(j,i) << std::endl;
                assert(0);
            }
            //std::cout << "SYSCONN: " << i << "," << j << "--" << sysConnIMtx(i,j) << std::endl;
            if (sysConnIMtx(i,j) > 0 && lnkScale < 0.9) {
                
                float scaledWt = (lnkScale * (sysConnIMtx(i,j) - minLnkWt)) + 1.0;
                sysConnMtx(i,j) = std::round(scaledWt);
            } else {
                sysConnMtx(i,j) = sysConnIMtx(i,j);
            }
            rowWt += sysConnMtx(i,j);
            ////std::cout << "SYSCONN_W: " << i << "," << j << "--" << sysConnMtx(i,j) << std::endl;
        }
        sysDegMtx(i,i) = rowWt;
    }

    ////std::cout << "SYS DEG: " << std::endl;
    ////std::cout << sysDegMtx << std::endl;

    MatrixXf sysLaplacianMtx = sysDegMtx - sysConnMtx;

    ////std::cout << "LAPL: " << std::endl;
    ////std::cout << sysLaplacianMtx << std::endl;
    for (unsigned i = 0; i < mNumFpgas; i++) {
        int rowSum = 0;
        for (unsigned j = 0; j < mNumFpgas; j++) {
            rowSum += sysLaplacianMtx(i,j);
        }
        assert(rowSum == 0);
    }
    
    MatrixXf cellWtMtxPowMinusHalf = MatrixXf::Zero(mNumFpgas, mNumFpgas);
    VectorXf cellInvSqrtDeg(mNumFpgas);
    for (unsigned i = 0; i < mNumFpgas; i++) {
        float deg = sysDegMtx(i, i);
        float invSqrt = deg > 0.0f ? (1.0f / std::sqrt(deg)) : 0.0f;
        cellWtMtxPowMinusHalf(i, i) = invSqrt;
        cellInvSqrtDeg[i] = invSqrt;
    }
    MatrixXf normSysLaplacianWtMtx = cellInvSqrtDeg.asDiagonal() * sysLaplacianMtx * cellInvSqrtDeg.asDiagonal();
    normSysLaplacianWtMtx = 0.5f * (normSysLaplacianWtMtx + normSysLaplacianWtMtx.transpose());

    VectorXf sysEigenVal(mNumFpgas);
    MatrixXf sysEigenVec(mNumFpgas, mNumFpgas);

    Eigen::SelfAdjointEigenSolver<MatrixXf> sysSAES(normSysLaplacianWtMtx);
    assert(sysSAES.info() == 0);
    sysEigenVal = sysSAES.eigenvalues();
    ////std::cout << "System Eigenvalues are: " << sysEigenVal << std::endl;
    sysEigenVec = cellInvSqrtDeg.asDiagonal() * sysSAES.eigenvectors();

    mSysEigenVec.resize(mNumFpgas, mNumFpgas);
    mSysEigenVec = sysEigenVec;

    // Eigen vectors corresponding to the first 2 non-zero eigen values
    // These correspond to x,y coordinate locations in the 2-D euclidean space
    VectorXf XV(mSysEigenVec.rows()),
             YV(mSysEigenVec.rows());
    // Get the eigen vectors for the first two eigen values greater than 0
    int minFpgaEIdx = 0;
    for (unsigned k = 0; k < sysEigenVal.size(); k++) {
        if (sysEigenVal[k] > 1e-4f) {
            minFpgaEIdx = k;
            break;
        }
    }
    if (minFpgaEIdx == 0)
        minFpgaEIdx = 1;
    assert(mNumFpgas > 2);// Atleast 3 FPGAs are needed to determine co smilarity
    assert(minFpgaEIdx > 0 && minFpgaEIdx < mNumFpgas); // Atleast one index must be a zero eigenval
    
    ////std::cout << "=====================================================" << std::endl;
    ////std::cout << "SYSTEM EIGEN VECTORS: " << std::endl;
    ////std::cout << sysEigenVec << std::endl;
    
    XV << sysEigenVec.col(minFpgaEIdx);
    YV << sysEigenVec.col(minFpgaEIdx+1);
    
    //std::cout << "EigenVectors are: " << XV << std::endl << std::endl << YV << std::endl;
    //std::cout << "System Eigen Vectors" << std::endl;
    std::ofstream ofs1("SEV.dat");
    for (int v1 = 0; v1 < XV.size(); v1++) {
        for (int v2 = 0; v2 < YV.size(); v2++) {
            if (v1 == v2)
                ofs1 << XV[v1] << "  " << YV[v1] << std::endl; 
        }
    }
    ofs1.close();

    // Now calculate the co similarity matrix

    int maxFpgaEIdx = mNumFpgas > 5  ? minFpgaEIdx + 4 : (mNumFpgas - 1);
    if (maxFpgaEIdx >= mNumFpgas)
        maxFpgaEIdx = mNumFpgas - 1;

    // Compute similarity for all FPGAs including unused
    MatrixXf coSimFpgas(mNumFpgas, mNumFpgas); // Each row of evec is for an fpga 
    coSimFpgas = MatrixXf::Zero(mNumFpgas, mNumFpgas);

    VectorXf vf1(sysEigenVec.cols());
    VectorXf vf2(sysEigenVec.cols());
    assert(vf1.size() > maxFpgaEIdx && vf2.size() > maxFpgaEIdx);
    VectorXf vf11 = VectorXf::Zero(maxFpgaEIdx);
    VectorXf vf22 = VectorXf::Zero(maxFpgaEIdx);

    const Matrix2D<unsigned>& sMtx = mFsdb->getSysMtx();
    double v1acc = 0.0, v2acc = 0.0, maxVal = MDBL_MIN;

    std::cout << "MIN/MAX fpga eigV index: " << minFpgaEIdx << "/" << maxFpgaEIdx << std::endl;
    std::pair<unsigned, unsigned> maxFIndx = std::make_pair(0,0);
    // Do this only if number of fpgas > 4.
    // For less than 4 fpgas, assume all are equally connected
    // and have same capacity.

    for (unsigned i = 0; i < mNumFpgas; i++) {
        bool iUnused = mFsdb->getCell(i).isUnused();
        vf1 = sysEigenVec.row(i);
        for (unsigned j = 0; j < mNumFpgas; j++) {
            if (i == j) {
                coSimFpgas(i,j) = 1.0;
                continue;
            }
            if (sMtx(i,j) == UINT_MAX) {
                continue;
            }
            bool jUnused = mFsdb->getCell(j).isUnused();
            if (iUnused || jUnused) {
                coSimFpgas(i,j) = -1.0;
                coSimFpgas(j,i) = -1.0;
                continue;
            }
            vf2 = sysEigenVec.row(j);
            v1acc = 0.0;
            v2acc = 0.0;
            vf11.setZero();
            vf22.setZero();
            for (int p = minFpgaEIdx, q = 0; p < maxFpgaEIdx; p++, q++) {
                vf11[q] = vf1[p+1];
                vf22[q] = vf2[p+1];
                v1acc += (vf11[q]*vf11[q]);
                v2acc += (vf22[q]*vf22[q]);
            }
            float v1sqrt = std::sqrt(v1acc);
            float v2sqrt = std::sqrt(v2acc);
            float prodMag = v1sqrt * v2sqrt;
            float dotProd = vf11.dot(vf22);
            float val = prodMag > 0.0f ? (dotProd / prodMag) : 0.0f;
            coSimFpgas(i,j) = val;
            if (i != j && val > maxVal) {
                maxVal = val;
                maxFIndx.first = i;
                maxFIndx.second = j;
            }
            ////std::cout << "COSIM FPGAs: " << i << "," << j << " :: " << val << std::endl; 
        }
    }
    std::cout << "MAX similarity is between fpgas: " << maxFIndx.first << "/" << maxFIndx.second
              << " - " << sMtx(maxFIndx.first, maxFIndx.second)
              << " - " << maxVal << std::endl;

    mCoSimFpgas.resize(mNumFpgas, mNumFpgas);
    mCoSimFpgas = coSimFpgas;
    ///////////////////////////////////////////////////////////////////////////
    // NETLIST
    const Matrix2D<unsigned>& instMtx = mNg->getInstMtx();
    MatrixXf netlistConnMtx(mNumInsts, mNumInsts);
    netlistConnMtx = MatrixXf::Zero(mNumInsts, mNumInsts);
    
    MatrixXf netlistDegMtx(mNumInsts, mNumInsts);
    netlistDegMtx = MatrixXf::Zero(mNumInsts, mNumInsts);

    MatrixXf instWtMtx(mNumInsts, mNumInsts);
    instWtMtx = MatrixXf::Zero(mNumInsts, mNumInsts);
    MatrixXf rawInstWtMtx(mNumInsts, mNumInsts);
    rawInstWtMtx = MatrixXf::Zero(mNumInsts, mNumInsts);

    // Accumulate all parallel nets between pairs of instances (u, v)
    const NetVec& allNets = mNg->getNets();
    for (const auto& nt : allNets) {
        const SourceDestInstPair& sdp = nt.getNetSrcDest();
        unsigned u = sdp.first;
        unsigned v = sdp.second;
        if (u != v && u < mNumInsts && v < mNumInsts) {
            float w = static_cast<float>(nt.getWeight());
            netlistConnMtx(u, v) += w;
            netlistConnMtx(v, u) += w;
        }
    }

    unsigned sparsity = 0;
    for (int i = 0; i < (int)mNumInsts; i++) {
        float rowWt = 0.0f;
        for (int j = 0; j < (int)mNumInsts; j++) {
            float wt = netlistConnMtx(i, j);
            if (i == j)
                assert(wt == 0.0f);
            if (wt == 0.0f)
                sparsity++;
            rowWt += wt;
        }
        netlistDegMtx(i,i) = rowWt;
        const NLInst& inst = mNg->getInst(i);
        unsigned instWt = inst.getLutWeight() + inst.getBMemWeight() + inst.getDMemWeight();
        // Some instances are not luts or memories -
        rawInstWtMtx(i,i) = instWt == 0 ? 1 : instWt;
        float sLInstWt = mNg->scaleInstLWt(inst.getLutWeight());
        float sBInstWt = mNg->scaleInstBWt(inst.getBMemWeight());
        float sDInstWt = mNg->scaleInstDWt(inst.getDMemWeight());
        
        float scaledInstWt = sLInstWt + sBInstWt + sDInstWt + 1.0;
        //std::cout << "scaled wts: " << sLInstWt << "/" << sBInstWt << "/" << sDInstWt << std::endl;

        if (instWt == 0) {
            std::cout << "WARNING: " << "Instance: " << i << " has zero wt - scaled wt: " << scaledInstWt << std::endl;
        }
        instWtMtx(i,i) = 1 + std::log(scaledInstWt);
        //std::cout << "INST-SWT: " << i << " = " << instWtMtx(i,i) << std::endl;
        if (std::isnan(instWtMtx(i,i)) || std::isinf(instWtMtx(i,i))) {
            std::cout << "FOUND inf or nan in inst wt: " << i << " -- " << scaledInstWt << std::endl;
            assert(0);
        }
    }

    // Now the instance weight matrix - diagonal matrix

#if 1 // Symmetricity check
    for (int i = 0; i < mNumInsts; i++) {
        for (int j = 0; j < mNumInsts; j++) {
            if (netlistConnMtx(i,j) != netlistConnMtx(j,i)) {
                std::cout << "NCONN NOT SYMM: " << i << "," << j << "--" << netlistConnMtx(i,j) << "/" << netlistConnMtx(j,i) << std::endl;
                assert(0);
            }
            //std::cout << "NCONN MTX: " << i << "," << j << "--" << netlistConnMtx(i,j) << std::endl;
        }
    }
#endif


    MatrixXf netlistLaplacianMtx = netlistDegMtx - netlistConnMtx;
    VectorXf instInvSqrtDeg(mNumInsts);
    for (int i = 0; i < (int)mNumInsts; i++) {
        float deg = netlistDegMtx(i, i);
        instInvSqrtDeg[i] = deg > 0.0f ? (1.0f / std::sqrt(deg)) : 0.0f;
    }
    MatrixXf normLaplacianW = instInvSqrtDeg.asDiagonal() * netlistLaplacianMtx * instInvSqrtDeg.asDiagonal();
    normLaplacianW = 0.5f * (normLaplacianW + normLaplacianW.transpose());
    double sty =  ((double)sparsity/(double)(mNumInsts*mNumInsts))*100.0;
    std::cout << "Sparsity is: " << sty <<  " percent" << std::endl;
    
    VectorXf nlEigenVal(mNumInsts);
    MatrixXf nlEigenVec(mNumInsts, mNumInsts);
    Eigen::SelfAdjointEigenSolver<MatrixXf> nlSAES(normLaplacianW);
    nlEigenVal = nlSAES.eigenvalues();
    std::cout << "EV computation: " << nlSAES.info() << std::endl;
    assert(nlSAES.info() == 0); // Success
    ////std::cout << "Netlist Eigenvalues are: " << nlEigenVal << std::endl;

    // Transform eigenvectors of D^{-1/2} L D^{-1/2} back to generalized eigenvectors D^{-1/2} U
    nlEigenVec = instInvSqrtDeg.asDiagonal() * nlSAES.eigenvectors();
    mNlEigenVec = nlEigenVec;
    VectorXf XNV(nlEigenVec.rows()),
             YNV(nlEigenVec.rows());

    // Skip trivial (near-zero) eigenvalues from disconnected or isolated vertices
    int minInstEIdx = 1;
    float zeroEigValVecMag = 1.0/(float)sqrt(mNumInsts);
    std::cout << "ZERO EIGVAL VEC MAG: " << zeroEigValVecMag << std::endl;
    for (int k = 1; k < nlEigenVal.size(); k++) {
        if (nlEigenVal[k] > 1e-4f) {
            minInstEIdx = k;
            break;
        }
    }

    assert(minInstEIdx < (nlEigenVal.size() - 1));
    XNV << nlEigenVec.col(minInstEIdx);
    YNV << nlEigenVec.col(minInstEIdx+1);
    ////std::cout << "EigenVectors are: " << XNV << std::endl << std::endl << YNV << std::endl;
    //std::cout << "Netlist Eigen Vectors" << std::endl;
    std::ofstream ofs("NEV.dat");
    for (int v1 = 0; v1 < XNV.size(); v1++) {
        for (int v2 = 0; v2 < YNV.size(); v2++) {
            if (v1 == v2)
                ofs << XNV[v1] << "  " << YNV[v1] << std::endl; 
        }
    }
    ofs.close();


    // Find median from XNV and bisect
    std::vector<float> xVals(XNV.size(), 0.0);
    for (unsigned v1 = 0; v1 < XNV.size(); v1++) {
        xVals[v1] = XNV[v1];
    }
    
    float median = findMedian(xVals);
    std::cout << "Median eigvec val is: " << median << std::endl;
    std::unordered_set<unsigned> cl1, cl2;
    unsigned wt1 = 0,
             wt2 = 0;
    for (unsigned v1 = 0; v1 < XNV.size(); v1++) {
        const NLInst& inst = mNg->getInst(v1);
        unsigned wt = inst.getLutWeight();
        if (XNV[v1] < median) {
            cl1.insert(v1);
            wt1 += wt;
        } else {
            cl2.insert(v1);
            wt2 += wt;
        }
    }

    std::cout << "cl1: " << cl1.size() << " -- cl2: " << cl2.size() << std::endl;
    size_t cutWt = 0;
    unsigned nrCutEdges = 0;
    const NetVec& ntV = mNg->getNets();
    for (auto& nt : ntV) {
        const Net& n = nt;
        const SourceDestInstPair& sdp = n.getNetSrcDest();
        unsigned i1 = sdp.first;
        unsigned i2 = sdp.second;
        if ((cl1.find(i1) != cl1.end() && cl2.find(i2) != cl2.end()) ||
                (cl1.find(i2) != cl1.end() && cl2.find(i1) != cl2.end())) {
            // cut edge
            cutWt += n.getWeight();
            nrCutEdges++;

        }
    }
    std::cout << "2-way Fiedler bisection cuts/wt: " << nrCutEdges << " -- " << cutWt <<std::endl;
    std::cout << "Cluster wts: " << wt1 << " -- " << wt2 <<std::endl;

    // Find prototype partition closest to coordinate 0,0. This is so that we can spread outwards
    // Rank remaining partitions as per the eq : cosine_similarity_to_last_prototype - remaining_links_to_last_prototype
    // Heapify partitions
    // Do the same for instances
    // Prototype instances are anchors in partitions.
   
    // Cosine similarities between instances with normalized vals of eigv
    // Cosine similarities between instances and fpgas with normalized vals of eigv
    // Cosine similarities between fpgas with normalized vals of eigv
   
    MatrixXf coSimInsts(mNumInsts, mNumInsts); // Each row of evec is for an inst
    coSimInsts = MatrixXf::Zero(mNumInsts, mNumInsts);

    int maxInstEIdx = std::min((int)nlEigenVec.cols() - 1, minInstEIdx + (int)mNumFpgas);
    std::cout << "min/max idx: " << minInstEIdx << "/" << maxInstEIdx << std::endl;
    VectorXf v1(nlEigenVec.cols());
    VectorXf v2(nlEigenVec.cols());
    assert(v1.size() > maxInstEIdx && v2.size() > maxInstEIdx);
    VectorXf v11 = VectorXf::Zero(maxInstEIdx);
    VectorXf v22 = VectorXf::Zero(maxInstEIdx);

    v1acc = 0.0, v2acc = 0.0, maxVal = MDBL_MIN;
    std::pair<unsigned, unsigned> maxIndx = std::make_pair(0,0);
    const Matrix2D<unsigned>& iMtx = mNg->getInstMtx();

    for (unsigned i = 0; i < mNumInsts; i++) {
        unsigned lwi = mNg->getInst(i).getLutWeight();
        v1 = nlEigenVec.row(i);
        for (unsigned j = 0; j < mNumInsts; j++) {
            if (i == j) {
                coSimInsts(i,j) = 1.0;
                coSimInsts(j,i) = 1.0;
                continue;
            }
            if (iMtx(i,j) == UINT_MAX) {
                coSimInsts(i,j) = -1.0;
                coSimInsts(j,i) = -1.0;
                continue;
            }
            unsigned lwj = mNg->getInst(j).getLutWeight();
            if  (lwi == 0 || lwj == 0) {
                coSimInsts(i,j) = -1.0;
                coSimInsts(j,i) = -1.0;
                continue;
            }
            v2 = nlEigenVec.row(j);
            v1acc = 0.0;
            v2acc = 0.0;
            v11.setZero();
            v22.setZero();
            for (int p = 0, q = minInstEIdx; q < maxInstEIdx; p++,q++) {
                v11[p] = v1[q];
                v22[p] = v2[q];
                v1acc += (v11[p]*v11[p]);
                v2acc += (v22[p]*v22[p]);
            }
            float v1sqrt = std::sqrt(v1acc);
            float v2sqrt = std::sqrt(v2acc);
            float prodMag = v1sqrt * v2sqrt;
            float dotProd = v11.dot(v22);
            float val = prodMag > 0.0f ? (dotProd / prodMag) : 0.0f;
            coSimInsts(i,j) = val;
            if (i != j && val > maxVal) {
                maxVal = val;
                maxIndx.first = i;
                maxIndx.second = j;
            }
        }
    }
    std::cout << "MAX similarity is between insts: " << maxIndx.first << "/" << maxIndx.second
              << " - " << iMtx(maxIndx.first, maxIndx.second)
              << " - " << maxVal << std::endl;
    mCoSimInsts.resize(mNumInsts, mNumInsts);
    mCoSimInsts = coSimInsts;

    // Now affinity between Instances and Fpgas
    MatrixXf coSimInstFpgas(mNumInsts, mNumFpgas); // Each row of evec is for an fpga 
    coSimInstFpgas = MatrixXf::Zero(mNumInsts, mNumFpgas);
    assert(sysEigenVec.cols() > 4);
    v1.resize(nlEigenVec.cols());
    v2.resize(sysEigenVec.cols());
    assert(v1.size() > 4 && v2.size() > 4);
    
    v1acc = 0.0, v2acc = 0.0, maxVal = MDBL_MIN;

    std::vector<int> fpga2Inst;
    std::vector<int> inst2Fpga;
    fpga2Inst.reserve(mNumFpgas);
    fpga2Inst.assign(mNumFpgas, 0);
    inst2Fpga.reserve(mNumInsts);
    inst2Fpga.assign(mNumInsts,0);

    std::unordered_set<int> cInsts;

    for (int i = 0; i < mNumInsts; i++) {
        double maxValForFpga = MDBL_MIN; 
        for (int j = 0; j < mNumFpgas; j++) {
            if (mFsdb->getCell(j).isUnused()) {
                coSimInstFpgas(i,j) = -1.0;
                continue;
            }
            if (mNg->getInst(i).getLutWeight() == 0) {
                coSimInstFpgas(i,j) = -1.0;
                continue;
            }
            v1 = nlEigenVec.row(i);
            v2 = sysEigenVec.row(j);
            v1acc = 0.0;
            v2acc = 0.0;
            v11.setZero();
            v22.setZero();
            for (int p = minInstEIdx, q = minFpgaEIdx, r = 0; q < maxFpgaEIdx; p++, q++, r++) {
                v11[r] = v1[p+1];
                v22[r] = v2[q+1];
                v1acc += (v11[r]*v11[r]);
                v2acc += (v22[r]*v22[r]);
            }
            float v1sqrt = std::sqrt(v1acc);
            float v2sqrt = std::sqrt(v2acc);
            float prodMag = v1sqrt * v2sqrt;
            float dotProd = v11.dot(v22);
            float val = prodMag > 0.0f ? (dotProd / prodMag) : 0.0f;
            coSimInstFpgas(i,j) = val;
            if (i != j) {
                if (val > maxVal) {
                    maxVal = val;
                    maxIndx.first = i;
                    maxIndx.second = j;
                }
                if (val > maxValForFpga) {
                    maxValForFpga = val;
                    inst2Fpga[i] = j;
                }
            }
        }
    }
    // From all insts for a FPGA, select one as centroid
    for (int i = 0; i < mNumFpgas; i++) {
        double maxVal = MDBL_MIN;
        for (int j = 0; j < mNumInsts; j++) {
            double val = coSimInstFpgas(j,i);
            if (cInsts.count(j) > 0)
                continue;
            if (val > maxVal) {
                maxVal = val;
                fpga2Inst[i] = j;
            }
        }
        cInsts.insert(fpga2Inst[i]);
    }
    std::cout << "MAX similarity is between Inst/Fpgas: " << maxIndx.first << "/" << maxIndx.second
              << " - " << maxVal << std::endl;
    mCoSimInstFpgas.resize(mNumInsts, mNumFpgas);
    mCoSimInstFpgas = coSimInstFpgas;

    // Compute clusters
    // Find eigen gap for instances
    std::vector<float> eigValDiff;
    int minEigId = minInstEIdx;
    int maxEigId = std::min((int)nlEigenVal.size() - 1, minEigId + (int)mNumFpgas);
    for (int i = minEigId; i <= maxEigId; i++) {
        eigValDiff.emplace_back(nlEigenVal[i]);
    }
    std::adjacent_difference(eigValDiff.begin(), eigValDiff.end(), eigValDiff.begin());
    std::pair<float,int> maxValIdx = std::make_pair(MFLT_MIN, 0);

    int maxDiffId = mNumFpgas > 12 ? mNumFpgas - 12 : 0;
    for (int j = (int)eigValDiff.size() - 1; j >= maxDiffId && j >= 0; j--) {
        if (eigValDiff[j] > maxValIdx.first) {
            maxValIdx.first = eigValDiff[j];
            maxValIdx.second = j;
        }
    }
    std::cout << "Eigen gap is: " << maxValIdx.second << "/" << maxValIdx.first << std::endl;

    size_t totalInstWt = rawInstWtMtx.trace();
    size_t totalFpgaWt = rawCellWtMtx.trace();
    std::cout << "Total Inst wt: " << totalInstWt << " - total cell wt: " << totalFpgaWt << std::endl;

    // Compute exact minimum number of FPGAs needed across LUT, BMEM, and DMEM capacities
    size_t totLutReq = 0, totBMemReq = 0, totDMemReq = 0;
    for (unsigned i = 0; i < mNumInsts; ++i) {
        const NLInst& inst = mNg->getInst(i);
        totLutReq += inst.getLutWeight();
        totBMemReq += inst.getBMemWeight();
        totDMemReq += inst.getDMemWeight();
    }
    unsigned fpgaLCap = std::max(1u, mFsdb->getCellLCap());
    unsigned fpgaBCap = std::max(1u, mFsdb->getCellBCap());
    unsigned fpgaDCap = std::max(1u, mFsdb->getCellDCap());
    int minFpgasL = std::ceil((double)totLutReq / (double)fpgaLCap);
    int minFpgasB = totBMemReq > 0 ? std::ceil((double)totBMemReq / (double)fpgaBCap) : 1;
    int minFpgasD = totDMemReq > 0 ? std::ceil((double)totDMemReq / (double)fpgaDCap) : 1;
    int lowerBoundIdx = std::max({2, minFpgasL, minFpgasB, minFpgasD});
    if (lowerBoundIdx > (int)mNumFpgas) {
        lowerBoundIdx = mNumFpgas;
    }
    std::cout << "Minimum number of FPGAs required: " << lowerBoundIdx << std::endl;
    assert(lowerBoundIdx > 0 && lowerBoundIdx <= (int)mNumFpgas);
    int idealIdx = maxValIdx.second > 2 ? maxValIdx.second : lowerBoundIdx;
    assert(idealIdx > 1);
    if (idealIdx < lowerBoundIdx) {
        idealIdx = lowerBoundIdx;
    }
    unsigned mNumInstsFit = mNumInsts/idealIdx;
    int avInstPerFpga = std::round((totalFpgaWt/mNumFpgas) / std::max<size_t>(1, totalInstWt/mNumInsts));
    double clusterScore = mNumInstsFit > (unsigned)avInstPerFpga ? 1.0 : ((idealIdx*100)/mNumFpgas);
    std::cout << "CLUSTERING SCORE (using graph eigengap): " << clusterScore << std::endl;

    // Use minimum number of FPGAs (lowerBoundIdx) as the target number of clusters
    // so we pack tightly without spreading across all FPGAs
    int numClusters = lowerBoundIdx;
    int numSpectralDims = std::min(numClusters, (int)nlEigenVec.cols() - minInstEIdx);
    assert(numSpectralDims > 0);
    MatrixXf reducedEV(mNumInsts, numSpectralDims);
    for (int c = 0; c < numSpectralDims; ++c) {
        float ev = std::max(nlEigenVal[minInstEIdx + c], 1e-4f);
        float weight = 1.0f / std::sqrt(ev);
        reducedEV.col(c) = nlEigenVec.col(minInstEIdx + c) * weight;
    }
    for (int r = 0; r < (int)mNumInsts; ++r) {
        float rowNorm = reducedEV.row(r).norm();
        if (rowNorm > 1e-12f) {
            reducedEV.row(r) /= rowNorm;
        }
    }

    assert(fpga2Inst.size() >= (size_t)numClusters);
    mClusters = KMeans::kmcluster(reducedEV, fpga2Inst, numClusters);
    
    unsigned csz = mClusters.size();
    mClusterSets.reserve(csz);
    for (unsigned i = 0; i < mClusters.size(); i++) {
        const std::vector<int>& clust = mClusters[i];
        std::unordered_set<unsigned> clustSet;
        //std::cout << "CLUSTER: " << i << " -- " << clust.size() << std::endl;
        for (unsigned j = 0; j < clust.size(); j++) {
            clustSet.insert(clust[j]);
        }
        mClusterSets.emplace_back(clustSet);
    }

}

void AnalyticalPartitioner::collectFpgaBridges()
{
    
    // Find multiple single-bridge FPGAs between unconnected pairs and sort them by weighted distance
    // TODO: Later find 2-leg and shorest path between unconnected pairs
    // Flattened vector of fpga-pairs
    // Connected pairs will have empty vector
    // Unconnected pairs with no bridge will have 1 element vector with UINT_MAX value
    //FpgaBridgeMtx bridgeFpgas(mNumFpgas*mNumFpgas, std::vector<unsigned>());
    mFpgaBridges.reserve(mNumFpgas*mNumFpgas);
    mFpgaBridges.assign(mNumFpgas*mNumFpgas, std::vector<unsigned>());

    for (unsigned i = 0; i < mNumFpgas; i++) {
        int numUnc = 0;
        for (unsigned j = 0; j < mNumFpgas; j++) {
            if (i == j)
                continue;
            if (mFsdb->isConnectedPair(i,j) == false) {
                numUnc++;
                int numBridges = 0;
                for (int k = 0; k < mNumFpgas; k++) {
                    if (k == j || k == i)
                        continue;
                    if (mFsdb->isConnectedPair(i,k) == false)
                        continue;
                    if (mFsdb->isConnectedPair(j,k) == false)
                        continue;
                    if (mFsdb->getBandwidth(i,k) == 0 || mFsdb->getBandwidth(j,k) == 0)
                        continue;
                    //std::cout << "\tFound bridge: " << k << std::endl;
                    //bridgeFpgas[i*mNumFpgas + j].emplace_back(k);
                    mFpgaBridges[i*mNumFpgas + j].emplace_back(k);
                    numBridges++;
                }
                assert(mFpgaBridges[i*mNumFpgas + j].size() == numBridges);
                //bridgeFpgas[i*mNumFpgas + j].emplace_back(UINT_MAX);
                mFpgaBridges[i*mNumFpgas + j].emplace_back(UINT_MAX);
                std::cout << "Unconnected pair: " << i << ", " << j << " : NUM BRDGs: "  << numBridges << std::endl;
            }
        }
        std::cout << "Num Unconnected for fpga: " << i << " : " << numUnc << std::endl;
    }
}

void AnalyticalPartitioner::getClusterOutNets(unsigned id, NetIdVec& nidv) const
{
    assert(id < mClusterSets.size()); 
    const std::unordered_set<unsigned>& clust = mClusterSets[id];
    size_t totOutNets = 0;
    unsigned csz = clust.size();
    //std::cout << "CLUSTER: " << i << " -- " << clust.size() << std::endl;
    unsigned totIsoInsts = 0;
    for (auto& inid : clust ) {
        const NLInst& nli = mNg->getInst(inid);
        const NetIdVec& nidV = nli.getNetsIds();
        unsigned totInstON = 0;
        for (auto& nid : nidV) {
            const Net& n = mNg->getConstNet(nid);
            const SourceDestInstPair& sd = n.getNetSrcDest();
            if (clust.find(sd.first) == clust.end() ||
                clust.find(sd.second) == clust.end()) {
                totOutNets++;
                nidv.emplace_back(nid);
                totInstON++;
            }
        }
        ////std::cout << "Inst out nets/totalNtets: " << inid << " / " << totInstON << " / " << nidV.size() << std::endl;
        if (totInstON == nidV.size())
            totIsoInsts++;
    }
    std::cout << "Number of cluster out nets/isoInsts: " << totOutNets << " / " << totIsoInsts <<  std::endl;

}

void AnalyticalPartitioner::updateInstAllocation(unsigned nId, unsigned pId)
{
    assert(pId != UINT_MAX);
    unsigned oldpId = mAllocatedInsts[nId];
    if (oldpId != pId && oldpId != UINT_MAX) {
        //std::cout << "Adding another part to inst: " << nId << " / " << pId << std::endl;
        removeInstFromPart(nId, oldpId);
    }
    mAllocatedInsts[nId] = pId;
    mFpgaNumInsts[pId]++;
    std::unordered_set<unsigned>& partInsts = mFpgaInsts[pId];
    partInsts.insert(nId);
}

void AnalyticalPartitioner::addToPartWts(unsigned pId, const CellData& cd)
{
    CellData& pW = getPartWeights(pId);
    std::get<0>(pW) = std::get<0>(cd);
    std::get<1>(pW) = std::get<1>(cd);
    std::get<2>(pW) = std::get<2>(cd);
}

bool AnalyticalPartitioner::checkAndAddInstToPart(unsigned nId, unsigned pId, float thr)
{
    const NLInst& nli = mNg->getInst(nId);
    const CellData& pW = getPartWeights(pId);
    unsigned cumLW = std::get<0>(pW) + nli.getLutWeight(); 
    unsigned cumBW = std::get<1>(pW) + nli.getBMemWeight(); 
    unsigned cumDW = std::get<2>(pW) + nli.getDMemWeight();
    const CellData& cumCd = std::make_tuple(cumLW, cumBW, cumDW);
    //
    if (!mFsdb->isCapViol(pId, cumCd, thr)) {
        //numInstsUnallocated--;
        updateInstAllocation(nId, pId);
        addToPartWts(pId, cumCd);
        return true;
    }
    return false;
}

bool AnalyticalPartitioner::canBeAddedToPart(unsigned nId, unsigned pId) const
{
    const NLInst& nli = mNg->getInst(nId);
    const CellData& pW = getPartWeights(pId);
    unsigned cumLW = std::get<0>(pW) + nli.getLutWeight(); 
    unsigned cumBW = std::get<1>(pW) + nli.getBMemWeight(); 
    unsigned cumDW = std::get<2>(pW) + nli.getDMemWeight();
    const CellData& cumCd = std::make_tuple(cumLW, cumBW, cumDW);
    //
    if (!mFsdb->isCapViol(pId, cumCd, 1.0)) {
        return true;
    }
    return false;
}

void AnalyticalPartitioner::removeInstFromPart(unsigned nId, unsigned pId)
{
    resetInstPart(nId);
    //numInstsUnallocated++;
    mFpgaNumInsts[pId]--;
    std::unordered_set<unsigned>& partInsts = mFpgaInsts[pId];
    std::unordered_set<unsigned>::iterator nbsit = partInsts.find(nId);
    if (nbsit != partInsts.end()) {
        partInsts.erase(nbsit); // Should not happen often
    }
    const NLInst& nbci = mNg->getInst(nId);
    CellData& nbWts = getPartWeights(pId);
    unsigned nbLwt = nbci.getLutWeight(),
             nbBwt = nbci.getBMemWeight(),
             nbDwt = nbci.getDMemWeight();
    std::get<0>(nbWts) -= nbLwt;
    std::get<1>(nbWts) -= nbBwt;
    std::get<2>(nbWts) -= nbDwt;
}

static RL::Problem buildRlProblem(const NetlistGraph& ng, const FpgaSystemDB& fsdb, const std::vector<unsigned>& parts)
{
    RL::Problem P;
    const unsigned n = ng.getNumInsts();
    const unsigned A = parts.size();
    P.n = n;
    P.A = A;
    P.off.assign(n + 1, 0);
    P.wl.resize(n);
    P.wb.resize(n);
    P.wd.resize(n);

    std::vector<double> accW(n, 0.0), accM(n, 0.0);
    std::vector<unsigned> nbrList;
    nbrList.reserve(256);
    std::vector<std::vector<std::pair<unsigned, std::pair<double, double>>>> rows(n);

    for (unsigned u = 0; u < n; ++u) {
        const NLInst& inst = ng.getInst(u);
        P.wl[u] = inst.getLutWeight();
        P.wb[u] = inst.getBMemWeight();
        P.wd[u] = inst.getDMemWeight();
        nbrList.clear();
        for (unsigned nid : inst.getNetsIds()) {
            const Net& net = ng.getConstNet(nid);
            const unsigned v = net.getNetOtherInst(u);
            if (v == u || v >= n)
                continue;
            if (accM[v] == 0.0)
                nbrList.push_back(v);
            accW[v] += static_cast<double>(net.getWeight());
            accM[v] += 1.0;
        }
        std::sort(nbrList.begin(), nbrList.end());
        rows[u].reserve(nbrList.size());
        for (unsigned v : nbrList) {
            rows[u].emplace_back(v, std::make_pair(accW[v], accM[v]));
            if (v > u) {
                P.totW += accW[v];
                P.totM += accM[v];
            }
            accW[v] = 0.0;
            accM[v] = 0.0;
        }
        P.off[u + 1] = P.off[u] + rows[u].size();
    }

    P.nbr.reserve(P.off[n]);
    P.wgt.reserve(P.off[n]);
    P.cnt.reserve(P.off[n]);
    for (unsigned u = 0; u < n; ++u) {
        for (const auto& e : rows[u]) {
            P.nbr.push_back(e.first);
            P.wgt.push_back(e.second.first);
            P.cnt.push_back(e.second.second);
        }
    }

    P.capL.resize(A);
    P.capB.resize(A);
    P.capD.resize(A);
    P.linked.assign((size_t)A * A, 0);
    for (unsigned a = 0; a < A; ++a) {
        const CellData& cd = fsdb.getPartCap(parts[a]);
        P.capL[a] = std::get<0>(cd);
        P.capB[a] = std::get<1>(cd);
        P.capD[a] = std::get<2>(cd);
        for (unsigned b = 0; b < A; ++b)
            P.linked[(size_t)a * A + b] = fsdb.isConnectedPair(parts[a], parts[b]) ? 1 : 0;
    }
    return P;
}

void AnalyticalPartitioner::initialPlacement()
{
    unsigned numClusters = mClusters.size();
    if (numClusters == 0 || mNumFpgas == 0)
        return;

    // 1. Build condensed instance adjacency list (merging parallel multi-edges)
    std::vector<std::vector<std::pair<unsigned, double>>> condensedAdj(mNumInsts);
    {
        std::vector<double> accumW(mNumInsts, 0.0);
        std::vector<unsigned> nbrList;
        nbrList.reserve(256);
        for (unsigned u = 0; u < mNumInsts; ++u) {
            nbrList.clear();
            const NetIdVec& incNets = mNg->getInst(u).getNetsIds();
            for (unsigned nid : incNets) {
                const Net& net = mNg->getConstNet(nid);
                unsigned v = net.getNetOtherInst(u);
                if (v == u || v >= mNumInsts) continue;
                if (accumW[v] == 0.0) {
                    nbrList.push_back(v);
                }
                accumW[v] += static_cast<double>(net.getWeight());
            }
            condensedAdj[u].reserve(nbrList.size());
            for (unsigned v : nbrList) {
                condensedAdj[u].emplace_back(v, accumW[v]);
                accumW[v] = 0.0;
            }
        }
    }

    // 2. Build cluster-to-cluster connectivity matrix
    std::vector<unsigned> instCluster(mNumInsts, 0);
    for (unsigned c = 0; c < numClusters; ++c) {
        for (int instId : mClusters[c]) {
            if (instId >= 0 && (unsigned)instId < mNumInsts) {
                instCluster[instId] = c;
            }
        }
    }

    MatrixXd clustConn = MatrixXd::Zero(numClusters, numClusters);
    for (unsigned u = 0; u < mNumInsts; ++u) {
        unsigned cu = instCluster[u];
        for (const auto& edge : condensedAdj[u]) {
            unsigned cv = instCluster[edge.first];
            if (cu != cv) {
                clustConn(cu, cv) += edge.second;
            }
        }
    }

    // 3. Build FPGA-to-FPGA penalty matrix (0 if same, 1.0 if connected, high penalty if unconnected)
    MatrixXd fpgaCost = MatrixXd::Zero(mNumFpgas, mNumFpgas);
    for (unsigned p1 = 0; p1 < mNumFpgas; ++p1) {
        for (unsigned p2 = 0; p2 < mNumFpgas; ++p2) {
            if (p1 == p2) {
                fpgaCost(p1, p2) = 0.0;
            } else if (mFsdb->isConnectedPair(p1, p2)) {
                unsigned bw = mFsdb->getBandwidth(p1, p2);
                fpgaCost(p1, p2) = 1.0 - (bw > 0 && bw != UINT_MAX ? std::min(0.2, (double)bw / 10000.0) : 0.0);
            } else {
                fpgaCost(p1, p2) = 50.0; // Heavy penalty for unconnected FPGA pairs
            }
        }
    }

    // 4. Select the best tightly-connected subset of active FPGAs of size numClusters,
    //    and optimize Cluster-to-FPGA mapping via 2-opt QAP search over all mNumFpgas
    std::vector<unsigned> clustToFpga(numClusters);
    {
        std::vector<bool> fpgaTaken(mNumFpgas, false);
        // Find seed FPGA with highest total link bandwidth to other active FPGAs
        unsigned bestSeed = 0;
        double bestSeedBw = -1.0;
        for (unsigned p = 0; p < mNumFpgas; ++p) {
            if (mFsdb->getCell(p).isUnused()) continue;
            double totBw = 0.0;
            for (unsigned q = 0; q < mNumFpgas; ++q) {
                if (p != q && mFsdb->isConnectedPair(p, q)) {
                    totBw += static_cast<double>(mFsdb->getBandwidth(p, q));
                }
            }
            if (totBw > bestSeedBw) {
                bestSeedBw = totBw;
                bestSeed = p;
            }
        }
        clustToFpga[0] = bestSeed;
        fpgaTaken[bestSeed] = true;

        // Greedily grow the subset of numClusters FPGAs that has maximum internal connectivity
        for (unsigned c = 1; c < numClusters; ++c) {
            unsigned bestP = UINT_MAX;
            double bestConnScore = std::numeric_limits<double>::lowest();
            for (unsigned p = 0; p < mNumFpgas; ++p) {
                if (fpgaTaken[p] || mFsdb->getCell(p).isUnused()) continue;
                double score = 0.0;
                for (unsigned prev = 0; prev < c; ++prev) {
                    unsigned qp = clustToFpga[prev];
                    if (mFsdb->isConnectedPair(p, qp)) {
                        score += 100.0 + 0.01 * static_cast<double>(mFsdb->getBandwidth(p, qp));
                    } else {
                        score -= 50.0;
                    }
                }
                if (score > bestConnScore) {
                    bestConnScore = score;
                    bestP = p;
                }
            }
            if (bestP == UINT_MAX) {
                for (unsigned p = 0; p < mNumFpgas; ++p) {
                    if (!fpgaTaken[p]) { bestP = p; break; }
                }
            }
            clustToFpga[c] = bestP;
            fpgaTaken[bestP] = true;
        }

        // 2-opt QAP swap search among mapped clusters and unselected candidate FPGAs
        bool improved = true;
        int qapIter = 0;
        while (improved && qapIter < 60) {
            improved = false;
            qapIter++;
            // Swaps between two assigned clusters
            for (unsigned c1 = 0; c1 < numClusters; ++c1) {
                for (unsigned c2 = c1 + 1; c2 < numClusters; ++c2) {
                    unsigned p1 = clustToFpga[c1];
                    unsigned p2 = clustToFpga[c2];
                    double delta = 0.0;
                    for (unsigned k = 0; k < numClusters; ++k) {
                        if (k == c1 || k == c2) continue;
                        unsigned pk = clustToFpga[k];
                        double w1k = clustConn(c1, k);
                        double w2k = clustConn(c2, k);
                        double oldCost = w1k * fpgaCost(p1, pk) + w2k * fpgaCost(p2, pk);
                        double newCost = w1k * fpgaCost(p2, pk) + w2k * fpgaCost(p1, pk);
                        delta += (newCost - oldCost);
                    }
                    if (delta < -1e-6) {
                        std::swap(clustToFpga[c1], clustToFpga[c2]);
                        improved = true;
                    }
                }
            }
            // Swaps between an assigned cluster c1 and an unassigned FPGA candP
            for (unsigned c1 = 0; c1 < numClusters; ++c1) {
                unsigned p1 = clustToFpga[c1];
                for (unsigned candP = 0; candP < mNumFpgas; ++candP) {
                    if (fpgaTaken[candP] || mFsdb->getCell(candP).isUnused()) continue;
                    double delta = 0.0;
                    for (unsigned k = 0; k < numClusters; ++k) {
                        if (k == c1) continue;
                        unsigned pk = clustToFpga[k];
                        double w1k = clustConn(c1, k);
                        delta += w1k * (fpgaCost(candP, pk) - fpgaCost(p1, pk));
                    }
                    if (delta < -1e-6) {
                        fpgaTaken[p1] = false;
                        fpgaTaken[candP] = true;
                        clustToFpga[c1] = candP;
                        p1 = candP;
                        improved = true;
                    }
                }
            }
        }
    }

    // Track active FPGAs (start with only the numClusters selected FPGAs; activate another only if strictly needed for capacity)
    std::vector<bool> activeFpgaMask(mNumFpgas, false);
    for (unsigned c = 0; c < numClusters; ++c) {
        activeFpgaMask[clustToFpga[c]] = true;
    }

    // 5. Capacity-aware initial assignment of instances from clusters to mapped FPGAs (using 100% usable capacity)
    std::vector<unsigned> overflowInsts;
    for (unsigned c = 0; c < numClusters; ++c) {
        unsigned targetP = clustToFpga[c];
        const std::vector<int>& clust = mClusters[c];
        for (int idx : clust) {
            unsigned instId = static_cast<unsigned>(idx);
            if (!checkAndAddInstToPart(instId, targetP, 1.0f)) {
                overflowInsts.push_back(instId);
            }
        }
    }

    // Sort overflow instances by total net weight descending so heavily connected instances get placed first
    std::sort(overflowInsts.begin(), overflowInsts.end(), [&](unsigned a, unsigned b) {
        return mNg->getInst(a).getTotalNetWt() > mNg->getInst(b).getTotalNetWt();
    });

    // Place any overflow instances into already-active FPGAs first; only open a new FPGA if no active FPGA can fit it
    for (unsigned instId : overflowInsts) {
        const NLInst& nli = mNg->getInst(instId);
        const auto& adj = condensedAdj[instId];

        auto findBestPart = [&](bool restrictToActive, float capThresh) -> unsigned {
            unsigned bestP = UINT_MAX;
            double bestScore = std::numeric_limits<double>::lowest();
            for (unsigned p = 0; p < mNumFpgas; ++p) {
                if (mFsdb->getCell(p).isUnused())
                    continue;
                if (restrictToActive && !activeFpgaMask[p])
                    continue;
                const CellData& pW = getPartWeights(p);
                CellData cumCd = std::make_tuple(
                    std::get<0>(pW) + nli.getLutWeight(),
                    std::get<1>(pW) + nli.getBMemWeight(),
                    std::get<2>(pW) + nli.getDMemWeight());
                if (mFsdb->isCapViol(p, cumCd, capThresh))
                    continue;

                double score = 0.0;
                for (const auto& edge : adj) {
                    unsigned otherP = getInstPart(edge.first);
                    if (otherP == UINT_MAX) continue;
                    double w = edge.second;
                    if (otherP == p) {
                        score += 2.0 * w;
                    } else if (mFsdb->isConnectedPair(p, otherP)) {
                        score += 1.0 * w;
                    } else {
                        score -= 4.0 * w;
                    }
                }
                // Bonus if p is connected to existing active FPGAs
                if (!restrictToActive) {
                    for (unsigned ap = 0; ap < mNumFpgas; ++ap) {
                        if (activeFpgaMask[ap] && mFsdb->isConnectedPair(p, ap)) {
                            score += 100.0;
                        }
                    }
                }

                if (score > bestScore) {
                    bestScore = score;
                    bestP = p;
                }
            }
            return bestP;
        };

        unsigned chosenP = findBestPart(true, 1.0f);
        if (chosenP == UINT_MAX) {
            chosenP = findBestPart(false, 1.0f);
            if (chosenP != UINT_MAX) {
                activeFpgaMask[chosenP] = true;
            }
        }
        if (chosenP == UINT_MAX) {
            chosenP = findBestPart(false, 1.5f);
            if (chosenP != UINT_MAX) {
                activeFpgaMask[chosenP] = true;
            }
        }
        if (chosenP == UINT_MAX) {
            chosenP = clustToFpga[0];
        }
        const CellData& pW = getPartWeights(chosenP);
        CellData cumCd = std::make_tuple(
            std::get<0>(pW) + nli.getLutWeight(),
            std::get<1>(pW) + nli.getBMemWeight(),
            std::get<2>(pW) + nli.getDMemWeight());
        updateInstAllocation(instId, chosenP);
        addToPartWts(chosenP, cumCd);
    }

    // 6. Fast Multi-pass k-way Fiduccia-Mattheyses (FM) Boundary Refinement
    //    Packs instances into the active FPGAs and minimizes total cut weight (cutWt + rtCutWt)
    std::vector<unsigned> instOrder(mNumInsts);
    for (unsigned i = 0; i < mNumInsts; ++i) {
        instOrder[i] = i;
    }
    std::sort(instOrder.begin(), instOrder.end(), [&](unsigned a, unsigned b) {
        return mNg->getInst(a).getTotalNetWt() > mNg->getInst(b).getTotalNetWt();
    });

    std::vector<double> partConnWts(mNumFpgas, 0.0);
    std::vector<unsigned> touchedParts;
    touchedParts.reserve(mNumFpgas);

    constexpr int maxRefinementPasses = 35;
    for (int pass = 0; pass < maxRefinementPasses; ++pass) {
        unsigned movesMade = 0;
        // Directly optimize total cut weight, with a tiny tie-breaker for connected links over route-throughs
        double rtPenalty = 1.0001;

        for (unsigned idx = 0; idx < mNumInsts; ++idx) {
            unsigned u = instOrder[idx];
            unsigned curP = getInstPart(u);
            if (curP == UINT_MAX)
                continue;

            const auto& adj = condensedAdj[u];
            if (adj.empty())
                continue;

            touchedParts.clear();
            bool hasExternalNeighbor = false;
            for (const auto& edge : adj) {
                unsigned pv = getInstPart(edge.first);
                if (pv == UINT_MAX)
                    continue;
                if (pv != curP)
                    hasExternalNeighbor = true;
                if (partConnWts[pv] == 0.0) {
                    touchedParts.push_back(pv);
                }
                partConnWts[pv] += edge.second;
            }

            if (!hasExternalNeighbor) {
                for (unsigned p : touchedParts) partConnWts[p] = 0.0;
                continue;
            }

            auto evalPartitionCost = [&](unsigned targetP) -> double {
                double cost = 0.0;
                for (unsigned pv : touchedParts) {
                    if (pv == targetP)
                        continue;
                    double w = partConnWts[pv];
                    if (mFsdb->isConnectedPair(targetP, pv)) {
                        cost += w;
                    } else {
                        cost += rtPenalty * w;
                    }
                }
                return cost;
            };

            double curCost = evalPartitionCost(curP);
            double bestCost = curCost;
            unsigned bestP = curP;

            for (unsigned candP : touchedParts) {
                if (candP == curP)
                    continue;
                if (!canBeAddedToPart(u, candP))
                    continue;
                double candCost = evalPartitionCost(candP);
                if (candCost < bestCost - 1e-6) {
                    bestCost = candCost;
                    bestP = candP;
                }
            }

            for (unsigned p : touchedParts) {
                partConnWts[p] = 0.0;
            }

            if (bestP != curP) {
                checkAndAddInstToPart(u, bestP, 1.0f);
                movesMade++;
            }
        }

        // Pairwise swap pass across top external candidate partitions when partitions are at 100% capacity
        if (pass < 20) {
            std::vector<std::pair<double, unsigned>> candQs;
            candQs.reserve(16);
            for (unsigned idx = 0; idx < mNumInsts; ++idx) {
                unsigned u = instOrder[idx];
                unsigned pu = getInstPart(u);
                if (pu == UINT_MAX) continue;
                const NLInst& instU = mNg->getInst(u);
                const auto& adjU = condensedAdj[u];

                touchedParts.clear();
                for (const auto& edge : adjU) {
                    unsigned pv = getInstPart(edge.first);
                    if (pv != UINT_MAX) {
                        if (partConnWts[pv] == 0.0) touchedParts.push_back(pv);
                        partConnWts[pv] += edge.second;
                    }
                }

                candQs.clear();
                for (unsigned q : touchedParts) {
                    if (q != pu && !canBeAddedToPart(u, q) && partConnWts[q] > 0.0) {
                        candQs.emplace_back(partConnWts[q], q);
                    }
                }
                for (unsigned q : touchedParts) partConnWts[q] = 0.0;

                if (candQs.empty())
                    continue;

                std::sort(candQs.begin(), candQs.end(), [](const std::pair<double, unsigned>& a, const std::pair<double, unsigned>& b) {
                    return a.first > b.first;
                });
                size_t maxCandQ = std::min<size_t>(candQs.size(), 3);

                unsigned bestV = UINT_MAX;
                unsigned bestQ = UINT_MAX;
                double bestSwapGain = 0.0;

                const CellData& wtP = getPartWeights(pu);

                for (size_t qi = 0; qi < maxCandQ; ++qi) {
                    unsigned targetQ = candQs[qi].second;
                    const std::unordered_set<unsigned>& qInsts = mFpgaInsts[targetQ];
                    const CellData& wtQ = getPartWeights(targetQ);

                    for (unsigned v : qInsts) {
                        const NLInst& instV = mNg->getInst(v);

                        if (std::get<0>(wtP) < instU.getLutWeight() ||
                            std::get<1>(wtP) < instU.getBMemWeight() ||
                            std::get<2>(wtP) < instU.getDMemWeight() ||
                            std::get<0>(wtQ) < instV.getLutWeight() ||
                            std::get<1>(wtQ) < instV.getBMemWeight() ||
                            std::get<2>(wtQ) < instV.getDMemWeight()) {
                            continue;
                        }
                        CellData newWtP = std::make_tuple(
                            std::get<0>(wtP) - instU.getLutWeight() + instV.getLutWeight(),
                            std::get<1>(wtP) - instU.getBMemWeight() + instV.getBMemWeight(),
                            std::get<2>(wtP) - instU.getDMemWeight() + instV.getDMemWeight());
                        CellData newWtQ = std::make_tuple(
                            std::get<0>(wtQ) - instV.getLutWeight() + instU.getLutWeight(),
                            std::get<1>(wtQ) - instV.getBMemWeight() + instU.getBMemWeight(),
                            std::get<2>(wtQ) - instV.getDMemWeight() + instU.getDMemWeight());

                        if (mFsdb->isCapViol(pu, newWtP, 1.0f) || mFsdb->isCapViol(targetQ, newWtQ, 1.0f))
                            continue;

                        double gain = 0.0;
                        for (const auto& edge : adjU) {
                            unsigned nbr = edge.first;
                            if (nbr == v) continue;
                            unsigned pnbr = getInstPart(nbr);
                            double w = edge.second;
                            double oldC = (pnbr == pu) ? 0.0 : (mFsdb->isConnectedPair(pu, pnbr) ? w : rtPenalty * w);
                            double newC = (pnbr == targetQ) ? 0.0 : (mFsdb->isConnectedPair(targetQ, pnbr) ? w : rtPenalty * w);
                            gain += (oldC - newC);
                        }
                        for (const auto& edge : condensedAdj[v]) {
                            unsigned nbr = edge.first;
                            if (nbr == u) continue;
                            unsigned pnbr = getInstPart(nbr);
                            double w = edge.second;
                            double oldC = (pnbr == targetQ) ? 0.0 : (mFsdb->isConnectedPair(targetQ, pnbr) ? w : rtPenalty * w);
                            double newC = (pnbr == pu) ? 0.0 : (mFsdb->isConnectedPair(pu, pnbr) ? w : rtPenalty * w);
                            gain += (oldC - newC);
                        }

                        if (gain > bestSwapGain + 1e-6) {
                            bestSwapGain = gain;
                            bestV = v;
                            bestQ = targetQ;
                        }
                    }
                }

                if (bestV != UINT_MAX && bestQ != UINT_MAX) {
                    removeInstFromPart(u, pu);
                    removeInstFromPart(bestV, bestQ);
                    checkAndAddInstToPart(u, bestQ, 1.0f);
                    checkAndAddInstToPart(bestV, pu, 1.0f);
                    movesMade += 2;
                }
            }
        }

        if (movesMade == 0 && pass >= 12)
            break;
    }

    // 7. Post-refinement Partition-to-FPGA QAP Remapping:
    //    Re-map the populated partitions onto the optimal physical FPGAs based on final cut traffic
    std::vector<unsigned> usedParts;
    for (unsigned p = 0; p < mNumFpgas; ++p) {
        if (mFpgaNumInsts[p] > 0) {
            usedParts.push_back(p);
        }
    }
    unsigned numUsed = usedParts.size();
    if (numUsed > 1) {
        std::vector<int> partToCompact(mNumFpgas, -1);
        for (unsigned i = 0; i < numUsed; ++i) {
            partToCompact[usedParts[i]] = i;
        }
        MatrixXd finalConn = MatrixXd::Zero(numUsed, numUsed);
        for (unsigned u = 0; u < mNumInsts; ++u) {
            unsigned pu = getInstPart(u);
            if (pu == UINT_MAX) continue;
            int ci = partToCompact[pu];
            if (ci < 0) continue;
            for (const auto& edge : condensedAdj[u]) {
                unsigned pv = getInstPart(edge.first);
                if (pv == UINT_MAX || pv == pu) continue;
                int cj = partToCompact[pv];
                if (cj >= 0) {
                    finalConn(ci, cj) += edge.second;
                }
            }
        }

        // Penalty matrix for final remapping: connected = 0.0, unconnected = 1.0 (to minimize rtCutWt without changing total cut)
        MatrixXd rtCostMtx = MatrixXd::Zero(mNumFpgas, mNumFpgas);
        for (unsigned p1 = 0; p1 < mNumFpgas; ++p1) {
            for (unsigned p2 = 0; p2 < mNumFpgas; ++p2) {
                if (p1 != p2 && !mFsdb->isConnectedPair(p1, p2)) {
                    rtCostMtx(p1, p2) = 1.0;
                }
            }
        }

        std::vector<unsigned> remapFpga = usedParts;
        std::vector<bool> fpgaUsedMask(mNumFpgas, false);
        for (unsigned p : remapFpga) fpgaUsedMask[p] = true;

        bool remapped = true;
        int remapIter = 0;
        while (remapped && remapIter < 50) {
            remapped = false;
            remapIter++;
            for (unsigned i = 0; i < numUsed; ++i) {
                for (unsigned j = i + 1; j < numUsed; ++j) {
                    unsigned p1 = remapFpga[i];
                    unsigned p2 = remapFpga[j];
                    // Check if p1 and p2 can accommodate each other's weights
                    if (mFsdb->isCapViol(p2, getPartWeights(usedParts[i]), 1.0f) ||
                        mFsdb->isCapViol(p1, getPartWeights(usedParts[j]), 1.0f)) {
                        continue;
                    }
                    double delta = 0.0;
                    for (unsigned k = 0; k < numUsed; ++k) {
                        if (k == i || k == j) continue;
                        unsigned pk = remapFpga[k];
                        delta += finalConn(i, k) * (rtCostMtx(p2, pk) - rtCostMtx(p1, pk))
                               + finalConn(j, k) * (rtCostMtx(p1, pk) - rtCostMtx(p2, pk));
                    }
                    if (delta < -1e-6) {
                        std::swap(remapFpga[i], remapFpga[j]);
                        remapped = true;
                    }
                }
            }
            for (unsigned i = 0; i < numUsed; ++i) {
                unsigned p1 = remapFpga[i];
                for (unsigned candP = 0; candP < mNumFpgas; ++candP) {
                    if (fpgaUsedMask[candP] || mFsdb->getCell(candP).isUnused()) continue;
                    if (mFsdb->isCapViol(candP, getPartWeights(usedParts[i]), 1.0f)) continue;
                    double delta = 0.0;
                    for (unsigned k = 0; k < numUsed; ++k) {
                        if (k == i) continue;
                        unsigned pk = remapFpga[k];
                        delta += finalConn(i, k) * (rtCostMtx(candP, pk) - rtCostMtx(p1, pk));
                    }
                    if (delta < -1e-6) {
                        fpgaUsedMask[p1] = false;
                        fpgaUsedMask[candP] = true;
                        remapFpga[i] = candP;
                        p1 = candP;
                        remapped = true;
                    }
                }
            }
        }

        // Apply the remapped FPGA IDs
        std::vector<unsigned> oldInstParts(mNumInsts);
        for (unsigned u = 0; u < mNumInsts; ++u) {
            oldInstParts[u] = getInstPart(u);
        }
        mFpgaNumInsts.assign(mNumFpgas, 0);
        mAllocatedInsts.assign(mNumInsts, UINT_MAX);
        mPartWeights.assign(mNumFpgas, std::make_tuple(0, 0, 0));
        mFpgaInsts.assign(mNumFpgas, std::unordered_set<unsigned>());

        for (unsigned u = 0; u < mNumInsts; ++u) {
            unsigned oldP = oldInstParts[u];
            if (oldP != UINT_MAX && partToCompact[oldP] >= 0) {
                unsigned newP = remapFpga[partToCompact[oldP]];
                const NLInst& nli = mNg->getInst(u);
                const CellData& pW = getPartWeights(newP);
                CellData cumCd = std::make_tuple(
                    std::get<0>(pW) + nli.getLutWeight(),
                    std::get<1>(pW) + nli.getBMemWeight(),
                    std::get<2>(pW) + nli.getDMemWeight());
                updateInstAllocation(u, newP);
                addToPartWts(newP, cumCd);
            }
        }
    }

    // 8. Reinforcement Learning (RL) Multi-Objective Post-Refinement:
    //    Uses a trained contextual policy to reduce total cut edges, total cut weight,
    //    and route-through edges simultaneously while preserving FPGA count and capacity limits.
    std::vector<unsigned> rlParts;
    for (unsigned p = 0; p < mNumFpgas; ++p) {
        if (mFpgaNumInsts[p] > 0)
            rlParts.push_back(p);
    }
    if (rlParts.size() > 1) {
        std::vector<int> rlCompact(mNumFpgas, -1);
        for (unsigned a = 0; a < rlParts.size(); ++a)
            rlCompact[rlParts[a]] = static_cast<int>(a);

        std::vector<unsigned> rlPa(mNumInsts, 0);
        bool allValid = true;
        for (unsigned u = 0; u < mNumInsts; ++u) {
            unsigned pu = getInstPart(u);
            if (pu == UINT_MAX || rlCompact[pu] < 0) {
                allValid = false;
                break;
            }
            rlPa[u] = static_cast<unsigned>(rlCompact[pu]);
        }

        if (allValid) {
            RL::Problem rlProb = buildRlProblem(*mNg, *mFsdb, rlParts);
            RL::Metrics mBefore, mAfter;
            RL::refinePartition(rlProb, rlPa, RL::activePolicy(), &mBefore, &mAfter);

            // Joint Partition-to-FPGA QAP Remapping + RL Second Pass:
            // Re-aligns refined partitions with physical FPGA links to further eliminate route-throughs
            const unsigned A = rlParts.size();
            std::vector<double> pairM((size_t)A * A, 0.0), pairW((size_t)A * A, 0.0);
            for (unsigned u = 0; u < mNumInsts; ++u) {
                const unsigned pu = rlPa[u];
                for (size_t idx = rlProb.off[u]; idx < rlProb.off[u + 1]; ++idx) {
                    const unsigned v = rlProb.nbr[idx];
                    const unsigned pv = rlPa[v];
                    if (pu != pv) {
                        pairM[(size_t)pu * A + pv] += rlProb.cnt[idx];
                        pairW[(size_t)pu * A + pv] += rlProb.wgt[idx];
                    }
                }
            }
            std::vector<unsigned> ldL(A, 0), ldB(A, 0), ldD(A, 0);
            for (unsigned u = 0; u < mNumInsts; ++u) {
                const unsigned a = rlPa[u];
                ldL[a] += rlProb.wl[u];
                ldB[a] += rlProb.wb[u];
                ldD[a] += rlProb.wd[u];
            }
            std::vector<bool> fpgaUsed(mNumFpgas, false);
            for (unsigned p : rlParts)
                fpgaUsed[p] = true;

            bool remappedAny = false;
            bool stepRemap = true;
            for (int iter = 0; stepRemap && iter < 40; ++iter) {
                stepRemap = false;
                for (unsigned i = 0; i < A; ++i) {
                    for (unsigned j = i + 1; j < A; ++j) {
                        const unsigned p1 = rlParts[i], p2 = rlParts[j];
                        if (mFsdb->isCapViol(p2, std::make_tuple(ldL[i], ldB[i], ldD[i]), 1.0f) ||
                            mFsdb->isCapViol(p1, std::make_tuple(ldL[j], ldB[j], ldD[j]), 1.0f))
                            continue;
                        double dRtM = 0.0, dRtW = 0.0;
                        for (unsigned k = 0; k < A; ++k) {
                            if (k == i || k == j)
                                continue;
                            const unsigned pk = rlParts[k];
                            const double rt1 = mFsdb->isConnectedPair(p1, pk) ? 0.0 : 1.0;
                            const double rt2 = mFsdb->isConnectedPair(p2, pk) ? 0.0 : 1.0;
                            dRtM += pairM[(size_t)i * A + k] * (rt2 - rt1) + pairM[(size_t)j * A + k] * (rt1 - rt2);
                            dRtW += pairW[(size_t)i * A + k] * (rt2 - rt1) + pairW[(size_t)j * A + k] * (rt1 - rt2);
                        }
                        if (dRtM < -1e-6 && dRtW <= 1e-6) {
                            std::swap(rlParts[i], rlParts[j]);
                            stepRemap = true;
                            remappedAny = true;
                        }
                    }
                }
                for (unsigned i = 0; i < A; ++i) {
                    unsigned p1 = rlParts[i];
                    for (unsigned candP = 0; candP < mNumFpgas; ++candP) {
                        if (fpgaUsed[candP] || mFsdb->getCell(candP).isUnused())
                            continue;
                        if (mFsdb->isCapViol(candP, std::make_tuple(ldL[i], ldB[i], ldD[i]), 1.0f))
                            continue;
                        double dRtM = 0.0, dRtW = 0.0;
                        for (unsigned k = 0; k < A; ++k) {
                            if (k == i)
                                continue;
                            const unsigned pk = rlParts[k];
                            const double rt1 = mFsdb->isConnectedPair(p1, pk) ? 0.0 : 1.0;
                            const double rtC = mFsdb->isConnectedPair(candP, pk) ? 0.0 : 1.0;
                            dRtM += pairM[(size_t)i * A + k] * (rtC - rt1);
                            dRtW += pairW[(size_t)i * A + k] * (rtC - rt1);
                        }
                        if (dRtM < -1e-6 && dRtW <= 1e-6) {
                            fpgaUsed[p1] = false;
                            fpgaUsed[candP] = true;
                            rlParts[i] = candP;
                            p1 = candP;
                            stepRemap = true;
                            remappedAny = true;
                        }
                    }
                }
            }

            if (remappedAny) {
                rlProb = buildRlProblem(*mNg, *mFsdb, rlParts);
                RL::refinePartition(rlProb, rlPa, RL::activePolicy(), nullptr, &mAfter, 2);
            }

            std::cout << "RL Refinement: cutEdges " << (size_t)mBefore.cutM << " -> " << (size_t)mAfter.cutM
                      << " | cutWt " << (size_t)mBefore.cutW << " -> " << (size_t)mAfter.cutW
                      << " | rtEdges " << (size_t)mBefore.rtM << " -> " << (size_t)mAfter.rtM
                      << " | rtWt " << (size_t)mBefore.rtW << " -> " << (size_t)mAfter.rtW << std::endl;

            mFpgaNumInsts.assign(mNumFpgas, 0);
            mAllocatedInsts.assign(mNumInsts, UINT_MAX);
            mPartWeights.assign(mNumFpgas, std::make_tuple(0, 0, 0));
            mFpgaInsts.assign(mNumFpgas, std::unordered_set<unsigned>());

            for (unsigned u = 0; u < mNumInsts; ++u) {
                unsigned newP = rlParts[rlPa[u]];
                const NLInst& nli = mNg->getInst(u);
                const CellData& pW = getPartWeights(newP);
                CellData cumCd = std::make_tuple(
                    std::get<0>(pW) + nli.getLutWeight(),
                    std::get<1>(pW) + nli.getBMemWeight(),
                    std::get<2>(pW) + nli.getDMemWeight());
                updateInstAllocation(u, newP);
                addToPartWts(newP, cumCd);
            }
        }
    }
}

void AnalyticalPartitioner::saveDBasMetisGraph()
{
    std::ofstream ofsIG("igraph.hgr");
    if (!ofsIG) {
        std::cout << "Cannot open file igraph.hgr to write." << std::endl;
        return;
    }
    
    int base = 0;
    ofsIG << "% num hyperedges, num vertices, format, 0/1-base\n";
    ofsIG << mNg->getNumNets() << ' ' << mNg->getNumInsts() << ' ' << "11" << ' ' << base << "\n";
    ofsIG << "% hyper edges\n";

    const NetVec& nv = mNg->getNets();
    const CellInstVec& iv = mNg->getInsts();

    for (auto& n : nv) {
        ofsIG << n.getWeight() << ' ';
        const SourceDestInstPair& sdp = n.getNetSrcDest();
        ofsIG << sdp.first + base << ' ';
        ofsIG << sdp.second + base << ' ';
        ofsIG << '\n';
    }

    size_t totInstsWt = 0;
    ofsIG << "% vertex weight\n";
    for (auto& i : iv) {
        unsigned wt = i.getLutWeight();
        ofsIG << wt << '\n';
        totInstsWt += wt;
    }
    
    ofsIG.close();

    std::ofstream ofsBG("blades.cons");
    if (!ofsBG) {
        std::cout << "Cannot open file blades.hgr to write." << std::endl;
        return;
    }
    // These are constraints in partitioners like HMETIS, FORGE etc
    // header:
    int ncons = 2; // BMEM and DMEM
    unsigned nparts = mFsdb->getNumFpgas();
    unsigned maxNetSize = 2; // GRAPH 
    int partitioningMode = 21; // UNDIRECTED_HYPERGRAPH_MODE

    size_t lbdCap = (size_t)mFsdb->getCellLCap() + (size_t)mFsdb->getCellBCap() + (size_t)mFsdb->getCellDCap();
    size_t totCap = lbdCap * (size_t)mNumFpgas;

    assert(totInstsWt > 0 && totInstsWt < totCap);
    double allowance = ((double)totCap/(double)totInstsWt) - 1.0;
    allowance = allowance > 0 ? allowance : 1.0;
    allowance *= 100.0;
    int bias = 0;

    ofsBG << "# constraint file: num_cons, num_parts, num_vertices, partitioning mode, allowance, bias, max net size\n";
    ofsBG << ncons << "\t" << nparts << "\t" << iv.size() << "\t" << partitioningMode
          << "\t" << allowance << "\t" << bias << "\t" << maxNetSize << "\n\n";
    ofsBG << "% connection scaling factors\n";
    for (unsigned i = 0; i < mNumFpgas; i++) {
        for (unsigned j = 0; j < mNumFpgas; j++) {
            bool isConn = mFsdb->isConnectedPair(i, j);
            unsigned bw = isConn ? mFsdb->getBandwidth(i, j) : 0;
            if (i == j || bw == UINT_MAX)
                bw = 0;
            ofsBG << bw << ' ';
        }
    }
   
    ofsBG << '\n';
    // constraint limits
    ofsBG << "# limits\n";

    // BMEM
    ofsBG << "BMEM" << "\t" << 1 << "\t";
    ofsBG << 0 << "\t";
    ofsBG << 0 << "\t";
    const FpgaCellVec& fpV = mFsdb->getCells();
    for (auto& fp : fpV) {
      unsigned bmems = fp.getNumBMems();
      ofsBG << bmems << '\t';
    }
    ofsBG << "\n";

    // DMEM
    ofsBG << "DMEM" << "\t" << 1 << "\t";
    ofsBG << 0 << "\t";
    ofsBG << 0 << "\t";
    for (auto& fp : fpV) {
      unsigned dmems = fp.getNumDMems();
      ofsBG << dmems << '\t';
    }
    ofsBG << "\n";

    ofsBG << "\n";
    
    // constraint values
    ofsBG << "# constraint values\n";
    for (auto& i : iv ) {
        for (unsigned c = 0; c < 2; c++) {
            unsigned wt = c == 0 ? i.getBMemWeight() : i.getDMemWeight();
            ofsBG << wt << '\t';
        }
        ofsBG << '\n';
    }
    ofsBG << "\n";

    ofsBG.close();
}

void AnalyticalPartitioner::readHMETISResults()
{
    std::ifstream resF("HMETIS.res");
    if (!resF.good()) {
        std::cout << "ERROR: File HMETIS.res: " << " is corrupt or unreadable" << std::endl;
        resF.close();
        assert(0);
    }

}

} // End Namespace SpectralSolver

using namespace SpectralSolver;

// Fast initial placement generator for RL training instances (selects active FPGAs, places via BFS + greedy FM)
static bool buildTrainingStart(const NetlistGraph& ng, const FpgaSystemDB& fsdb, RL::TrainInstance& outInst)
{
    const unsigned n = ng.getNumInsts();
    const unsigned nf = fsdb.getNumFpgas();
    double totL = 0, totB = 0, totD = 0;
    for (unsigned u = 0; u < n; ++u) {
        totL += ng.getInst(u).getLutWeight();
        totB += ng.getInst(u).getBMemWeight();
        totD += ng.getInst(u).getDMemWeight();
    }
    unsigned k = std::max<unsigned>(2u, std::max({
        (unsigned)std::ceil(totL / std::max(1u, fsdb.getCellLCap())),
        (unsigned)std::ceil(totB / std::max(1u, fsdb.getCellBCap())),
        (unsigned)std::ceil(totD / std::max(1u, fsdb.getCellDCap()))
    }));
    if (k > nf)
        k = nf;

    // Grow a tightly-connected set of active FPGAs
    auto selectParts = [&](unsigned targetK) {
        std::vector<unsigned> parts;
        std::vector<char> taken(nf, 0);
        unsigned bestSeed = UINT_MAX;
        double bestBw = -1.0;
        for (unsigned p = 0; p < nf; ++p) {
            if (fsdb.getCell(p).isUnused())
                continue;
            double bw = 0.0;
            for (unsigned q = 0; q < nf; ++q)
                if (q != p && fsdb.isConnectedPair(p, q))
                    bw += fsdb.getBandwidth(p, q);
            if (bw > bestBw) {
                bestBw = bw;
                bestSeed = p;
            }
        }
        if (bestSeed == UINT_MAX)
            return parts;
        parts.push_back(bestSeed);
        taken[bestSeed] = 1;
        while (parts.size() < targetK) {
            unsigned bestP = UINT_MAX;
            double bestScore = -1e300;
            for (unsigned p = 0; p < nf; ++p) {
                if (taken[p] || fsdb.getCell(p).isUnused())
                    continue;
                double s = 0.0;
                for (unsigned q : parts)
                    s += fsdb.isConnectedPair(p, q) ? (100.0 + 0.01 * fsdb.getBandwidth(p, q)) : -50.0;
                if (s > bestScore) {
                    bestScore = s;
                    bestP = p;
                }
            }
            if (bestP == UINT_MAX)
                break;
            parts.push_back(bestP);
            taken[bestP] = 1;
        }
        return parts;
    };

    for (unsigned extra = 0; extra <= std::max(4u, k / 5); ++extra) {
        std::vector<unsigned> parts = selectParts(std::min(nf, k + extra));
        if (parts.size() < 2)
            continue;
        RL::Problem P = buildRlProblem(ng, fsdb, parts);
        // BFS ordering from highest-weight vertex
        std::vector<double> deg(n, 0.0);
        for (unsigned u = 0; u < n; ++u)
            for (size_t idx = P.off[u]; idx < P.off[u + 1]; ++idx)
                deg[u] += P.wgt[idx];
        std::vector<unsigned> roots(n);
        std::iota(roots.begin(), roots.end(), 0u);
        std::sort(roots.begin(), roots.end(), [&](unsigned a, unsigned b) {
            return deg[a] != deg[b] ? deg[a] > deg[b] : a < b;
        });
        std::vector<unsigned> order;
        order.reserve(n);
        std::vector<char> vis(n, 0);
        for (unsigned r : roots) {
            if (vis[r])
                continue;
            vis[r] = 1;
            std::vector<unsigned> q{r};
            for (size_t h = 0; h < q.size(); ++h) {
                unsigned u = q[h];
                order.push_back(u);
                for (size_t idx = P.off[u]; idx < P.off[u + 1]; ++idx) {
                    unsigned v = P.nbr[idx];
                    if (!vis[v]) {
                        vis[v] = 1;
                        q.push_back(v);
                    }
                }
            }
        }

        const unsigned A = P.A;
        std::vector<unsigned> pa(n, UINT_MAX), ldL(A, 0), ldB(A, 0), ldD(A, 0);
        bool allPlaced = true;
        unsigned curA = 0;
        for (unsigned u : order) {
            bool placed = false;
            for (unsigned tries = 0; tries < A; ++tries) {
                unsigned a = (curA + tries) % A;
                if (ldL[a] + P.wl[u] <= P.capL[a] &&
                    ldB[a] + P.wb[u] <= P.capB[a] &&
                    ldD[a] + P.wd[u] <= P.capD[a]) {
                    pa[u] = a;
                    ldL[a] += P.wl[u];
                    ldB[a] += P.wb[u];
                    ldD[a] += P.wd[u];
                    if (ldL[a] * 10 >= P.capL[a] * 9)
                        curA = (a + 1) % A;
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                allPlaced = false;
                break;
            }
        }
        if (!allPlaced)
            continue;

        // Run a fast greedy 1-opt warm-up on cut weight so training starts from a realistic local minimum
        RL::RolloutEnv st(P, pa);
        for (int pass = 0; pass < 10; ++pass) {
            unsigned moves = 0;
            for (unsigned u : order) {
                unsigned pu = st.pa[u];
                double bestDW = 0.0;
                unsigned bestQ = UINT_MAX;
                for (unsigned q = 0; q < A; ++q) {
                    if (q == pu || !st.canMove(u, q))
                        continue;
                    double dW, dM, dRtM, dRtW;
                    st.singleDeltas(u, q, dW, dM, dRtM, dRtW);
                    if (dW > bestDW + 1e-6) {
                        bestDW = dW;
                        bestQ = q;
                    }
                }
                if (bestQ != UINT_MAX) {
                    st.applyMove(u, bestQ);
                    ++moves;
                }
            }
            if (moves == 0)
                break;
        }
        outInst.P = std::move(P);
        outInst.initPa = std::move(st.pa);
        return true;
    }
    return false;
}

// CLI mode: spec_part --rl-train [policy_out] [epochs] [lr] <graph_dir1> <graph_dir2> ...
static int runRlTrainingCli(int argc, char** argv)
{
    std::string outPath = (argc > 2) ? argv[2] : "rl_policy.weights";
    int epochs = (argc > 3) ? std::atoi(argv[3]) : 60;
    double lr = (argc > 4) ? std::atof(argv[4]) : 0.08;

    std::vector<RL::TrainInstance> corpus;
    for (int i = 5; i < argc; ++i) {
        std::string dir = argv[i];
        LinkDataMap lm;
        CellDataVec cv;
        InstDataVec iv;
        NetDataVec nv;
        SysDefData sd = std::make_tuple(0, 0, 0, 0, 0);
        JsonReader jr;
        jr.readFile(dir + "/blades.json", dir + "/igraph.json", lm, cv, sd, iv, nv);
        FpgaSystemDB fdb(lm, cv, sd);
        NetlistGraph ng(nv, iv, false);

        RL::TrainInstance inst;
        inst.name = dir;
        if (buildTrainingStart(ng, fdb, inst)) {
            std::cout << "Loaded RL training instance: " << dir
                      << " (insts=" << inst.P.n << ", activeFPGAs=" << inst.P.A << ")" << std::endl;
            corpus.push_back(std::move(inst));
        }
    }
    if (corpus.empty()) {
        std::cout << "ERROR: No valid training graphs loaded." << std::endl;
        return 1;
    }
    RL::trainPolicyCorpus(corpus, epochs, lr, outPath);
    std::cout << "Saved trained RL policy to: " << outPath << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--rl-train") {
        return runRlTrainingCli(argc, argv);
    }
    if (argc > 3) {
        RL::policyFilePath() = argv[3];
    }

    LinkDataMap linkDataMap;
    CellDataVec cellDatV;
    InstDataVec iData;
    NetDataVec nData;

    SysDefData sysDat = std::make_tuple(0,0,0,0,0);

    std::string bladesPath = (argc > 1) ? argv[1] : "./blades.json";
    std::string igraphPath = (argc > 2) ? argv[2] : "./igraph.json";

    JsonReader jr;
    jr.readFile(bladesPath, igraphPath, linkDataMap, cellDatV, sysDat, iData, nData);

#if 0 // DEBUG
    //iData and nData
    for (auto& idt : iData) {
        const InstData& el = idt;
        std::cout << "Inst: " << std::get<0>(el) << std::endl;
        const NetIdVec& nv = std::get<1>(el);
        for (auto& nt : nv) {
            std::cout << "\t";
            std::cout << "NET: " << nt << std::endl;
        }
        std::cout << "\t" << "LUTs: " << std::get<2>(el) << "/BMEMs: " << std::get<3>(el) << "/DMEMs: " << std::get<4>(el) << std::endl;
    }

    for (auto& ndt : nData) {
        const NetData& el = ndt;
        std::cout << "Net: " << std::get<0>(el) << "/" << std::get<1>(el).first << "--" << std::get<1>(el).second << "/" << std::get<2>(el) << std::endl;
    }
#endif

    bool unw = false; // Unit Net Weight mode
    FpgaSystemDB fpdb(linkDataMap, cellDatV, sysDat);
    NetlistGraph ng(nData, iData, unw);
    
    // FPGA SYSTEM
    unsigned numFpgas = fpdb.getNumFpgas();
    unsigned numInsts = ng.getNumInsts();

    // Setup the placer
    AnalyticalPartitioner ap(ng, fpdb, false);
    ap.saveDBasMetisGraph(); // FOR HMETIS
    
    const Matrix2D<unsigned>& sysMtx = fpdb.getSysMtx();
    
    const MatrixXf& coSimInsts = ap.getCoSimInsts();
    const MatrixXf& coSimFpgas = ap.getCoSimFpgas();
    const MatrixXf& coSimInstFpgas = ap.getCoSimInstFpgas();

    const ClusterId2Clust& clusters = ap.getClusters();
    
    unsigned clSz = clusters.size();

    // DUMP
    for (unsigned i = 0; i < clSz; i++) {
        NetIdVec nidv;
        unsigned clustSz = clusters[i].size();
        std::cout << "===============================" << std::endl;
        std::cout << "CLUST - num insts: " << i << " / " << clustSz << std::endl; 
        nidv.reserve(clustSz+1u);
        ap.getClusterOutNets(i, nidv);
        std::cout << "CLUST - num out nets: " << i << " / " << nidv.size() << std::endl;
        std::cout << "===============================" << std::endl;
    }

#if 1
    //// SUMMARY REPORT
    int fpgaN = 0;
    int totalInsts = 0;
    const PartNumInstsVec& fni = ap.getPartInsts();
    for (auto& eli : fni) {
        std::cout << "Num Insts: " << fpgaN << " : " << eli << std::endl;
        totalInsts += eli;
        fpgaN++;
    }

    size_t cutWt = 0;
    size_t rtCutWt = 0;
    unsigned nrtCutEdges = 0;
    unsigned rtCutEdges = 0;
    const NetVec& ntV = ng.getNets();
    Matrix2D<unsigned> rtMtx(numFpgas, numFpgas);
    for (auto& nt : ntV) {
        const Net& n = nt;
        const SourceDestInstPair& sdp = n.getNetSrcDest();
        unsigned i1 = sdp.first;
        unsigned i2 = sdp.second;
        unsigned p1 = ap.getInstPart(i1);
        unsigned p2 = ap.getInstPart(i2);
        if (p1 != UINT_MAX && p2 != UINT_MAX && p1 != p2) {
            if (fpdb.isConnectedPair(p1,p2)) {
                cutWt += n.getWeight();
                nrtCutEdges++;
            } else {
                rtCutWt += n.getWeight();
                rtCutEdges++;
                unsigned preVal = rtMtx(p1,p2);
                rtMtx(p1, p2, preVal + 1);
                //std::cout << "UNC-RT: " << p1 << " -- " << p2 << std::endl;
            }
        }
    }
    // Reverse lookup fpga pair
    // Generate move repository
    std::vector<std::pair<unsigned, unsigned>> fpgaPairs;
    std::multiset<std::pair<unsigned, unsigned>, RankComp> rtPairSet;
    for (unsigned i = 0; i < numFpgas; i++) {
        unsigned iVal = i*numFpgas;
        for (unsigned j = 0; j < numFpgas; j++) {
            fpgaPairs.emplace_back(std::make_pair(i,j));
            unsigned pairId = iVal + j;
            rtPairSet.insert(std::make_pair(pairId, rtMtx(i,j)));
            if (rtMtx(i,j) > 0) {
                ////std::cout << "RT: " << i << "," << j << "/" << rtMtx(i,j) << std::endl;
            }
        }
    }

    for (auto& el : rtPairSet) {
        const std::pair<unsigned, unsigned>& fpgaPairRT = el;
        if (fpgaPairRT.second > 0) {
            std::cout << fpgaPairs[fpgaPairRT.first].first << "," << fpgaPairs[fpgaPairRT.first].second << " --- " << fpgaPairRT.second << std::endl; 
        }
    }
    std::cout << "Total Insts allocated: " << totalInsts << std::endl;
    std::cout << "Total cut edges/weight: " << nrtCutEdges << " / " << cutWt << std::endl;
    std::cout << "Total rt cut edges/weight: " << rtCutEdges << " / " << rtCutWt << std::endl;

    std::string partFileName = "igraph.hgr.part." + std::to_string(numFpgas) + ".0";
    std::ofstream ofsPart(partFileName);
    if (ofsPart) {
        for (unsigned i = 0; i < numInsts; i++) {
            ofsPart << ap.getInstPart(i) << "\n";
        }
        ofsPart.close();
    }
    std::ofstream ofsRes(partFileName + ".result");
    if (ofsRes) {
        ofsRes << (cutWt + rtCutWt) << "\n";
        ofsRes.close();
    }
  
#endif
    
    return 0;
}
