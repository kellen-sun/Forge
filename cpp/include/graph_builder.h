#pragma once

#include "ir.h"

class GraphBuilder {
   private:
    IR& ir_;

   public:
    explicit GraphBuilder(IR& ir) : ir_(ir) {}

    int append(Node node);
    void replaceUses(int old_id, int new_id);
    void replace(int old_id, int new_id);
    void setOutput(int node_id);
};
