#include "dosling_scaler.h"
#include "student-v37.h"
int main(void) {
  dl_context *context =
      dl_create_from_memory(dos_lut_student_v37, dos_lut_student_v37_size);
  if (!context)
    return 1;
  uint8_t source[3] = {17, 91, 233}, output[27];
  int status = dl_scale3(context, source, 1, 1, 3, output, 9, 0, 0, 0);
  dl_destroy(context);
  if (status)
    return 2;
  for (int i = 0; i < 9; i++)
    for (int k = 0; k < 3; k++)
      if (output[i * 3 + k] != source[k])
        return 3;
  return 0;
}
