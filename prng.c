#if 0
#!/bin/bash
(set -x; gcc -Wall -O2 -msse2 prng.c -o prng)
echo "Use it like this:" 
echo "dd if=<(./prng) ..."
echo "shred --random-source=<(./prng) ..."
exit
#endif

/*
This program is meant for shredding fast SSDs quickly
on my system /dev/urandom was 600 MB/s but this program is 8 GB/s
*/

#include <stdio.h>
#include <unistd.h>
#include <xmmintrin.h>
#include <stdint.h>

static void write_random( size_t bufsize, void *buffer, void *state_p )
{
	__m128i state = _mm_load_si128( state_p );
	const size_t bs = 16;
	const __m128i a=_mm_set1_epi16( 27893 ), c=_mm_set1_epi16( 7777 );

	for(size_t off=0; off<bufsize; off+=bs) {
		__m128i x = state;
		/* Basic 16-bit linear congruential generator. Kinda bad but also very very fast */
		x = _mm_mullo_epi16( a, x );
		x = _mm_add_epi16( c, x );
		state = x;
		_mm_store_si128( (void*)((char*) buffer + off), x );
	}

	_mm_store_si128( state_p, state );
}

int main()
{
	char buf[64*1024];
	const size_t bufsize = sizeof(buf);
	char state[16];

	FILE *f=fopen("/dev/urandom", "r");
	if (!f) {
		perror("failed to open /dev/urandom");
		return 1;
	}
	if (fread(state, 1, sizeof(state), f) != sizeof(state)) {
		perror("failed to read /dev/urandom");
		return 1;
	}
	fclose(f);

	for(;;) {
		write_random(bufsize, buf, state);
		if (write(1, buf, bufsize) != bufsize) {
			break;
		}
	}

	return 0;
}

