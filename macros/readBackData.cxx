#include <Entities/Vertex.h>
#include <ecs/Entity.h>
#include <ecs/EntityCollection.h>
using namespace o2;
using namespace ecs;
using namespace vertex;
int readNewEvents(const char *file) {
  using Entity_t = Vertex<ESDEventMapping>;
  Handler h;
  // tell the handler to read all the data from the file we just made.
  // it will only read the first page in memory to read all metadata,
  // afterwards it lazilly loads the components it needs by using mmap
  // (pruning)
  h.readFile(file);

  /// Create an EntityCollection from our handler. This will fetch data from
  /// the handler and fail if components are missing
  EntityCollection<Entity_t> container(h);
  unsigned event_id = 0;
  unsigned event_start = 0;
  std::cout << container.size() << std::endl;
  for (unsigned u = 0; u < container.size(); u++) {
    auto map = container[u].get<ESDEventMapping>();
    std::cout << u << "\t" << map.trackCount() << std::endl;
  }

  return 0;
}

#ifndef __CINT__
int main(int argc, char **argv) { return readNewEvents(argv[1]); }
#endif
