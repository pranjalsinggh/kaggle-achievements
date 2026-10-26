// Original Connect X search implementation. MIT license.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

using U64 = uint64_t;
using Clock = std::chrono::steady_clock;
static constexpr int order[7] = {3,2,4,1,5,0,6};
static constexpr U64 bottom = (1ULL<<0)|(1ULL<<7)|(1ULL<<14)|(1ULL<<21)|(1ULL<<28)|(1ULL<<35)|(1ULL<<42);
static constexpr U64 boardmask = bottom*63;
static constexpr U64 center = 63ULL<<21;
static constexpr int bits = 21;
struct Entry { U64 key=0; int16_t score=0; int8_t depth=-1; uint8_t flag=0; int8_t move=3; };
static std::vector<Entry> table(1<<bits);
static std::vector<U64> lines;
static Clock::time_point deadline;
static uint64_t nodes=0;
static bool stopped=false;
static int completed_depth=0;

static int pop(U64 p) { return __builtin_popcountll(p); }
static U64 winning(U64 p,U64 occupied) {
    U64 r=(p<<1)&(p<<2)&(p<<3);
    for(int d : {6,7,8}) {
        r|=(p<<d)&(p<<(2*d))&(p<<(3*d));
        r|=(p>>d)&(p>>(2*d))&(p>>(3*d));
        r|=(p<<d)&(p>>d)&(p>>(2*d));
        r|=(p<<(2*d))&(p<<d)&(p>>d);
    }
    return r&(boardmask^occupied);
}

static void initialize() {
    if(!lines.empty()) return;
    const int dirs[4][2]={{0,1},{1,0},{1,1},{1,-1}};
    for(int r=0;r<6;r++) for(int c=0;c<7;c++) for(auto &d:dirs) {
        if(r+3*d[0]<0||r+3*d[0]>=6||c+3*d[1]<0||c+3*d[1]>=7) continue;
        U64 mask=0;
        for(int i=0;i<4;i++) mask |= 1ULL<<(7*(c+i*d[1])+r+i*d[0]);
        lines.push_back(mask);
    }
}

static int evaluate(U64 pos,U64 mask) {
    U64 opp=pos^mask;
    int score=3*(pop(pos&center)-pop(opp&center));
    const int weights[]={0,1,6,30,1000};
    for(U64 line:lines) {
        U64 a=pos&line,b=opp&line;
        if(!b) score+=weights[pop(a)];
        else if(!a) score-=weights[pop(b)];
    }
    score+=12*(pop(winning(pos,mask))-pop(winning(opp,mask)));
    return score;
}

struct Move { U64 bit; int col; int priority; };

static int search(U64 pos,U64 mask,int stones,int depth,int alpha,int beta) {
    if((++nodes&2047)==0 && Clock::now()>=deadline) stopped=true;
    if(stopped) return 0;
    U64 possible=(mask+bottom)&boardmask;
    if(winning(pos,mask)&possible) return 10000-stones;
    if(stones>=41) return 0;
    U64 threats=winning(pos^mask,mask);
    U64 forced=possible&threats;
    if(forced) {
        if(forced&(forced-1)) return -10000+stones+1;
        possible=forced;
    }
    possible &= ~(threats>>1);
    if(!possible) return -10000+stones+1;
    if(depth==0) return evaluate(pos,mask);
    U64 key=pos+mask+1;
    auto slot=(key*11400714819323198485ULL)>>(64-bits);
    Entry cached=table[slot];
    int preferred=-1;
    if(cached.key==key) {
        preferred=cached.move;
        if(cached.depth>=depth) {
            if(cached.flag==1) return cached.score;
            if(cached.flag==2) alpha=std::max(alpha,int(cached.score));
            else if(cached.flag==3) beta=std::min(beta,int(cached.score));
            if(alpha>=beta) return cached.score;
        }
    }
    int original_alpha=alpha,original_beta=beta;
    std::array<Move,7> moves;
    int count=0;
    for(int c:order) {
        U64 bit=possible&(63ULL<<(7*c));
        if(!bit) continue;
        int priority=(c==preferred?100000:0)+100*pop(winning(pos|bit,mask|bit))+3-std::abs(3-c);
        int i=count++;
        while(i>0 && priority>moves[i-1].priority) { moves[i]=moves[i-1];i--; }
        moves[i]={bit,c,priority};
    }
    int best=-20000,best_col=3;
    for(int i=0;i<count;i++) {
        int score=-search(pos^mask,mask|moves[i].bit,stones+1,depth-1,-beta,-alpha);
        if(stopped) return 0;
        if(score>best) { best=score;best_col=moves[i].col; }
        alpha=std::max(alpha,score);
        if(alpha>=beta) break;
    }
    Entry fresh;
    fresh.key=key;fresh.score=int16_t(best);fresh.depth=int8_t(depth);fresh.move=int8_t(best_col);
    fresh.flag=best<=original_alpha?3:(best>=original_beta?2:1);
    table[slot]=fresh;
    return best;
}

EXPORT int cx_choose(const int *board,int mark,double seconds,int max_depth) {
    initialize();
    U64 pos=0,mask=0;
    int stones=0;
    for(int r=0;r<6;r++) for(int c=0;c<7;c++) if(board[r*7+c]) {
        U64 bit=1ULL<<(7*c+5-r);
        mask|=bit;stones++;
        if(board[r*7+c]==mark) pos|=bit;
    }
    U64 possible=(mask+bottom)&boardmask;
    U64 wins=winning(pos,mask)&possible;
    for(int c:order) if(wins&(63ULL<<(7*c))) return c;
    U64 threats=winning(pos^mask,mask);
    U64 forced=possible&threats;
    if(forced && !(forced&(forced-1))) {
        for(int c:order) if(forced&(63ULL<<(7*c))) return c;
    }
    U64 safe=(forced?forced:possible)&~(threats>>1);
    std::vector<int> candidates;
    for(int c:order) if(safe&(63ULL<<(7*c))) candidates.push_back(c);
    if(candidates.empty()) {
        for(int c:order) if(board[c]==0) return c;
        return 0;
    }
    if(candidates.size()==1) return candidates[0];
    if(stones==0) return 3;
    stopped=false;nodes=0;completed_depth=0;
    deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    int best=candidates[0];
    std::array<int,7> scores{};
    for(int depth=1;depth<=std::min(max_depth,42-stones);depth++) {
        int alpha=-20000,iteration_best=best;
        for(int c:candidates) {
            U64 bit=possible&(63ULL<<(7*c));
            int score=-search(pos^mask,mask|bit,stones+1,depth-1,-20000,-alpha);
            if(stopped) break;
            scores[c]=score;
            if(score>alpha) { alpha=score;iteration_best=c; }
        }
        if(stopped) break;
        best=iteration_best;completed_depth=depth;
        std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b) { return scores[a]>scores[b]; });
        if(std::abs(alpha)>9000 || Clock::now()>=deadline) break;
    }
    return best;
}

EXPORT int cx_depth() { return completed_depth; }
EXPORT uint64_t cx_nodes() { return nodes; }
EXPORT U64 cx_winning(U64 p,U64 mask) { return winning(p,mask); }
EXPORT void cx_reset() { std::fill(table.begin(),table.end(),Entry{}); }
