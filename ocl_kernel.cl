// Minimal OpenCL kernels for Keyhunt

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uint k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline uint ROTR(uint x, uint n){ return (x>>n)|(x<<(32-n)); }
inline uint CH(uint x,uint y,uint z){ return (x & y) ^ (~x & z); }
inline uint MAJ(uint x,uint y,uint z){ return (x & y) ^ (x & z) ^ (y & z); }
inline uint BSIG0(uint x){ return ROTR(x,2) ^ ROTR(x,13) ^ ROTR(x,22); }
inline uint BSIG1(uint x){ return ROTR(x,6) ^ ROTR(x,11) ^ ROTR(x,25); }
inline uint SSIG0(uint x){ return ROTR(x,7) ^ ROTR(x,18) ^ (x>>3); }
inline uint SSIG1(uint x){ return ROTR(x,17) ^ ROTR(x,19) ^ (x>>10); }

__kernel void sha256_33_kernel(__global const uchar *in, __global uchar *out){
    int gid = get_global_id(0);
    const __global uchar *d = in + gid*64;
    __global uchar *o = out + gid*32;

    uint w[64];
    for(int i=0;i<16;i++){
        w[i] = ((uint)d[i*4]<<24)|((uint)d[i*4+1]<<16)|((uint)d[i*4+2]<<8)|((uint)d[i*4+3]);
    }
    for(int i=16;i<64;i++)
        w[i] = SSIG1(w[i-2]) + w[i-7] + SSIG0(w[i-15]) + w[i-16];

    uint a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,dv=0xa54ff53a,
         e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;

    for(int i=0;i<64;i++){
        uint t1 = h + BSIG1(e) + CH(e,f,g) + k[i] + w[i];
        uint t2 = BSIG0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=dv + t1; dv=c; c=b; b=a; a=t1+t2;
    }

    uint res[8];
    res[0]=a+0x6a09e667; res[1]=b+0xbb67ae85; res[2]=c+0x3c6ef372; res[3]=dv+0xa54ff53a;
    res[4]=e+0x510e527f; res[5]=f+0x9b05688c; res[6]=g+0x1f83d9ab; res[7]=h+0x5be0cd19;

    for(int i=0;i<8;i++){
        o[i*4+0]=(uchar)(res[i]>>24);
        o[i*4+1]=(uchar)(res[i]>>16);
        o[i*4+2]=(uchar)(res[i]>>8);
        o[i*4+3]=(uchar)(res[i]);
    }
}

/* RIPEMD160 constants */
__constant uint rmd_k1[5] = {0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
__constant uint rmd_k2[5] = {0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

inline uint ROL(uint x, uint n){ return (x<<n)|(x>>(32-n)); }
inline uint f1(uint x,uint y,uint z){ return x ^ y ^ z; }
inline uint f2(uint x,uint y,uint z){ return (x & y) | (~x & z); }
inline uint f3(uint x,uint y,uint z){ return (x | ~y) ^ z; }
inline uint f4(uint x,uint y,uint z){ return (x & z) | (y & ~z); }
inline uint f5(uint x,uint y,uint z){ return x ^ (y | ~z); }

__kernel void ripemd160_32_kernel(__global const uchar *in, __global uchar *out){
    int gid = get_global_id(0);
    const __global uchar *d = in + gid*64;
    __global uchar *o = out + gid*20;

    uint w[16];
    for(int i=0;i<16;i++)
        w[i]= ((uint)d[i*4]<<24)|((uint)d[i*4+1]<<16)|((uint)d[i*4+2]<<8)|((uint)d[i*4+3]);

    uint a1=0x67452301,b1=0xEFCDAB89,c1=0x98BADCFE,d1=0x10325476,e1=0xC3D2E1F0;
    uint a2=a1,b2=b1,c2=c1,d2=d1,e2=e1;

    int s1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6,
        8,13,6,5,15,13,11,11};
    int s2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11,
        9,7,15,11,8,6,6,14};
    int r1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13,
        5,1,3,7,14,6,9,11};
    int r2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11,
        0,8,7,6,4,2,1,13};

    for(int i=0;i<80;i++){
        uint f,t;
        if(i<16){ f=f1(b1,c1,d1); t=rmd_k1[0]; }
        else if(i<32){ f=f2(b1,c1,d1); t=rmd_k1[1]; }
        else if(i<48){ f=f3(b1,c1,d1); t=rmd_k1[2]; }
        else if(i<64){ f=f4(b1,c1,d1); t=rmd_k1[3]; }
        else { f=f5(b1,c1,d1); t=rmd_k1[4]; }
        uint temp=ROL(a1+f+w[r1[i]]+t,s1[i])+e1; a1=e1; e1=d1; d1=ROL(c1,10); c1=b1; b1=temp;

        if(i<16){ f=f5(b2,c2,d2); t=rmd_k2[0]; }
        else if(i<32){ f=f4(b2,c2,d2); t=rmd_k2[1]; }
        else if(i<48){ f=f3(b2,c2,d2); t=rmd_k2[2]; }
        else if(i<64){ f=f2(b2,c2,d2); t=rmd_k2[3]; }
        else { f=f1(b2,c2,d2); t=rmd_k2[4]; }
        temp=ROL(a2+f+w[r2[i]]+t,s2[i])+e2; a2=e2; e2=d2; d2=ROL(c2,10); c2=b2; b2=temp;
    }

    uint t=d1+c2+0x10325476;
    c2=c1+d2+0x98BADCFE;
    c1=b1+e2+0xEFCDAB89;
    b1=a1+a2+0x67452301;
    a1=t;
    uint digest[5]={a1,b1,c1,c2,e1+d2+0xC3D2E1F0};

    for(int i=0;i<5;i++){
        o[i*4]=(uchar)(digest[i]);
        o[i*4+1]=(uchar)(digest[i]>>8);
        o[i*4+2]=(uchar)(digest[i]>>16);
        o[i*4+3]=(uchar)(digest[i]>>24);
    }
}
