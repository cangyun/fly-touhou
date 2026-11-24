#include <gum/guminterceptor.h>

typedef struct {
    double minimum;
    guint enabled;
    guint owner_thread;
} BranchLink;

typedef struct {
    double margin;
    gboolean captured;
} CollisionCall;

enum CollisionKind {
    BULLET = 1,
    ENEMY_BODY = 2,
    LASER = 3
};

extern BranchLink th08_branch;
extern guint8 th08_player[];
extern double branch_cos(double angle);
extern double branch_sin(double angle);

static double max_gap(double x, double y)
{
    return x != x ? x : y != y ? y : x > y ? x : y;
}

static double absolute(double x)
{
    return x < 0. ? -x : x;
}

static double laser_margin(GumInvocationContext *context, const float *center, const float *size)
{
    const float *origin = gum_invocation_context_get_nth_argument(context, 2);
    const float *position = (const float *) (th08_player + 0x2b4);
    const float *hurt_half = (const float *) (th08_player + 0x3d4);
    union {
        guint32 bits;
        float value;
    } rotation;
    rotation.bits = GPOINTER_TO_UINT(gum_invocation_context_get_nth_argument(context, 3));
    const double cos_angle = branch_cos(rotation.value);
    const double sin_angle = branch_sin(rotation.value);
    const double dx = (double) position[0] - origin[0];
    const double dy = (double) position[1] - origin[1];
    const double x = absolute(dx * cos_angle + dy * sin_angle - ((double) center[0] - origin[0]));
    const double y = absolute(dy * cos_angle - dx * sin_angle - ((double) center[1] - origin[1]));
    return max_gap(x - .5 * size[0] - hurt_half[0], y - .5 * size[1] - hurt_half[1]);
}

static double box_margin(const float *center, const float *size)
{
    const float *low = (const float *) (th08_player + 0x38c);
    const float *high = (const float *) (th08_player + 0x398);
    const double half_x = .5 * size[0];
    const double half_y = .5 * size[1];
    const double x = max_gap(center[0] - half_x - high[0], low[0] - (center[0] + half_x));
    const double y = max_gap(center[1] - half_y - high[1], low[1] - (center[1] + half_y));
    return max_gap(x, y);
}

void enter_collision(GumInvocationContext *context)
{
    if (!th08_branch.enabled || gum_invocation_context_get_thread_id(context) != th08_branch.owner_thread)
        return;
    CollisionCall *call = GUM_IC_GET_INVOCATION_DATA(context, CollisionCall);
    const guint kind = GPOINTER_TO_UINT(gum_invocation_context_get_listener_function_data(context));
    const float *center = gum_invocation_context_get_nth_argument(context, 0);
    const float *size = gum_invocation_context_get_nth_argument(context, 1);
    call->captured = TRUE;
    call->margin = kind == LASER ? laser_margin(context, center, size) : box_margin(center, size);
}

void leave_collision(GumInvocationContext *context)
{
    if (!th08_branch.enabled || gum_invocation_context_get_thread_id(context) != th08_branch.owner_thread)
        return;
    const CollisionCall *call = GUM_IC_GET_INVOCATION_DATA(context, CollisionCall);
    const guint kind = GPOINTER_TO_UINT(gum_invocation_context_get_listener_function_data(context));
    const guint result = GPOINTER_TO_UINT(gum_invocation_context_get_return_value(context));
    if (!call->captured || (kind == BULLET && result == 2))
        return;
    if (call->margin < th08_branch.minimum || call->margin != call->margin)
        th08_branch.minimum = call->margin;
}
