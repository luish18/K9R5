/* Newlib syscall stubs backed by GVSoC semihosting. */
#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

#include "semihost.h"

extern char __heap_start[];
extern char __heap_end[];

void *_sbrk(int incr) {
  static char *brk = __heap_start;
  if (brk + incr > __heap_end) {
    errno = ENOMEM;
    return (void *)-1;
  }
  char *prev = brk;
  brk += incr;
  return prev;
}

int _write(int fd, const char *buf, int len) {
  (void)fd;
  for (int i = 0; i < len; i++)
    sh_putc(buf[i]);
  return len;
}

int _read(int fd, char *buf, int len) {
  (void)fd;
  (void)buf;
  (void)len;
  return 0;
}

int _close(int fd) {
  (void)fd;
  return -1;
}

int _fstat(int fd, struct stat *st) {
  (void)fd;
  st->st_mode = S_IFCHR;
  return 0;
}

int _isatty(int fd) {
  (void)fd;
  return 1;
}

int _lseek(int fd, int offset, int whence) {
  (void)fd;
  (void)offset;
  (void)whence;
  return 0;
}

void _exit(int code) { sh_exit(code); }

int _kill(int pid, int sig) {
  (void)pid;
  (void)sig;
  errno = EINVAL;
  return -1;
}

int _getpid(void) { return 1; }
